# -*- coding: utf-8 -*-
"""
memory.py —— 永久对话记忆层（SQLite + WAL，标准库 sqlite3，零额外依赖）
====================================================================
表结构：
  messages(id, agent_name, role, content, created_at, session_id, archived)
  summaries(id, agent_name, scope, content, created_at)   -- scope: long_term / screen

铁律：
  * 屏幕图像永不入库，只存文本
  * 记忆按 agent_name 隔离；share_memory=true 时 long_term 摘要跨智能体共享
  * 所有读写都有异常保护：SQLite 出错只记日志，绝不影响语音主流程
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from typing import List, Optional, Tuple

log = logging.getLogger("mimo.memory")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  agent_name TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at REAL NOT NULL,
  session_id TEXT,
  archived INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_msg_agent ON messages(agent_name, archived, id);
CREATE TABLE IF NOT EXISTS summaries(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  agent_name TEXT NOT NULL,
  scope TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sum_agent ON summaries(agent_name, scope, id);
"""


def fmt_ts(epoch: float) -> str:
    try:
        return time.strftime("%H:%M", time.localtime(epoch))
    except Exception:
        return "??:??"


class MemoryStore:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        # check_same_thread=False + 自带 RLock：主线程和后台摘要线程共用一个连接，
        # 比每线程一个连接更简单，且所有操作都串行化，不会踩 SQLite 的线程限制。
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")   # WAL：读写并发更好，断电更安全
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        log.info("记忆库已就绪: %s", db_path)

    # ------------------------------------------------------------ 写入
    def add_message(self, agent: str, role: str, content: str, session_id: str = "") -> None:
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO messages(agent_name, role, content, created_at, session_id, archived)"
                    " VALUES(?,?,?,?,?,0)",
                    (agent, role, content, time.time(), session_id))
                self._conn.commit()
        except Exception as e:
            log.error("消息入库失败（不影响语音回复）: %s", e)

    def add_summary(self, agent: str, scope: str, content: str) -> None:
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO summaries(agent_name, scope, content, created_at) VALUES(?,?,?,?)",
                    (agent, scope, content, time.time()))
                self._conn.commit()
            log.info("已写入 %s 摘要: %s...", scope, content[:40])
        except Exception as e:
            log.error("摘要入库失败（不影响语音回复）: %s", e)

    # ------------------------------------------------------------ 读取
    def recent_messages(self, agent: str, limit: int = 20) -> List[dict]:
        """启动/每轮对话时加载的短期上下文：最近 N 条、未被摘要归档的消息。"""
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT role, content FROM messages WHERE agent_name=? AND archived=0"
                    " ORDER BY id DESC LIMIT ?", (agent, limit)).fetchall()
            return [{"role": r, "content": c} for r, c in reversed(rows)]
        except Exception as e:
            log.error("读取历史失败: %s", e)
            return []

    def count_unarchived(self, agent: str) -> int:
        try:
            with self._lock:
                n = self._conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE agent_name=? AND archived=0",
                    (agent,)).fetchone()[0]
            return int(n or 0)
        except Exception as e:
            log.error("统计消息失败: %s", e)
            return 0

    def pending_summary_rows(self, agent: str, limit: int) -> List[Tuple[int, str, str]]:
        """较早的、还没被压缩归档的消息（从最旧开始取一批，供长期摘要用）。"""
        try:
            with self._lock:
                return self._conn.execute(
                    "SELECT id, role, content FROM messages WHERE agent_name=? AND archived=0"
                    " ORDER BY id ASC LIMIT ?", (agent, limit)).fetchall()
        except Exception as e:
            log.error("读取待摘要消息失败: %s", e)
            return []

    def mark_archived(self, ids: List[int]) -> None:
        try:
            with self._lock:
                self._conn.executemany("UPDATE messages SET archived=1 WHERE id=?",
                                       [(i,) for i in ids])
                self._conn.commit()
        except Exception as e:
            log.error("标记归档失败: %s", e)

    def latest_long_term(self, agent: str, shared: bool = False) -> Optional[str]:
        """最新一条长期记忆摘要。shared=True 时跨智能体共享（share_memory 配置）。"""
        try:
            with self._lock:
                if shared:
                    row = self._conn.execute(
                        "SELECT content FROM summaries WHERE scope='long_term'"
                        " ORDER BY id DESC LIMIT 1").fetchone()
                else:
                    row = self._conn.execute(
                        "SELECT content FROM summaries WHERE scope='long_term' AND agent_name=?"
                        " ORDER BY id DESC LIMIT 1", (agent,)).fetchone()
            return row[0] if row else None
        except Exception as e:
            log.error("读取长期摘要失败: %s", e)
            return None

    def recent_screen_summaries(self, agent: str, max_age_minutes: float,
                                limit: int = 2) -> List[Tuple[str, float]]:
        """近期屏幕记忆：只取最新几条且未过期的（过期的不再注入 system prompt，省 token）。"""
        try:
            cutoff = time.time() - max_age_minutes * 60
            with self._lock:
                rows = self._conn.execute(
                    "SELECT content, created_at FROM summaries"
                    " WHERE agent_name=? AND scope='screen' AND created_at>=?"
                    " ORDER BY id DESC LIMIT ?", (agent, cutoff, limit)).fetchall()
            return [(c, t) for c, t in rows]
        except Exception as e:
            log.error("读取屏幕摘要失败: %s", e)
            return []

    # ------------------------------------------------------------ 维护
    def reset(self) -> None:
        """清空所有智能体的对话与摘要（--reset-memory，主脚本已做二次确认）。"""
        with self._lock:
            self._conn.execute("DELETE FROM messages")
            self._conn.execute("DELETE FROM summaries")
            self._conn.commit()
        log.warning("记忆库已清空")

    def close(self) -> None:
        try:
            with self._lock:
                self._conn.commit()
                self._conn.close()
        except Exception:
            pass
