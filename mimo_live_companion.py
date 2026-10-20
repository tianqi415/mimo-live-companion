#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mimo_live_companion.py —— 小米 MIMO 实时语音游戏陪玩（Windows 10/11, Python 3.10+）
==================================================================================
麦克风实时对话 + 游戏画面感知 + 永久记忆 + 智能体切换，全程走小米 MIMO API。

【启动步骤】
  1) 安装依赖:  pip install -r requirements.txt
  2) 复制 .env.example 为 .env，填入 MIMO_API_KEY=...（不填会中文提示后退出）
  3) 复制 config.example.yaml 为 config.yaml，按需修改（游戏窗口关键字、VAD、截图、记忆参数）
  4) 智能体: 复制 agents/example.yaml 为 agents/你的名字.yaml，按注释改成你自己的人设
  5) 运行:     python mimo_live_companion.py --agent example   （example 换成你的智能体名）
  建议戴耳机：扬声器外放会被麦克风收到，容易误触发"插话打断"。

【运行中命令】（控制台直接输入）
  agent <名字>   切换智能体（持久化记忆不丢失）
  agents         列出全部智能体
  status         查看当前状态
  quit / exit    退出（Ctrl+C 同样可以）
  也可按 F8 在智能体之间循环切换（keyboard 库偶尔需要管理员权限，失败请用控制台命令）

【记忆】
  位置: data/memory.db（SQLite + WAL；屏幕图片永不入库）
  新增智能体: 在 agents/ 下新建 xxx.yaml，然后 --agent xxx 或运行中 agent xxx
  清空记忆:   python mimo_live_companion.py --reset-memory（输入 yes 二次确认）

【做成配置项的不确定项（均有日志提示，不会静默失败）】
  - config.yaml → tts.sample_rate：流式 PCM16 采样率默认 24000；若音调/语速不对就改它
  - capturekit 的窗口抓帧 API 若与你安装版本不一致：打日志并自动回退 mss
  - TTS 非流式回退的音频字段做了多路径提取，失败会明确报错（把日志发我即可调整）
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import queue
import signal
import sys
import threading
import time
import uuid
from collections import deque
from typing import Optional

import numpy as np
import sounddevice as sd
import yaml
from dotenv import load_dotenv

from mimo_api import MimoApiError, MimoClient, pcm16_to_wav_bytes, wav_to_pcm16
from memory import MemoryStore, fmt_ts

log = logging.getLogger("mimo")

# flash 失败时的固定兜底话术（按需求：走 TTS 播报后回空闲）
FALLBACK_REPLY = "我这边网络有点延迟，你刚才说什么？"
# 用户主动要求总结屏幕的关键词（触发立即做一次屏幕阶段总结）
SUMMARY_KEYWORDS = ("总结一下", "总结下", "复盘", "刚才怎么样", "刚打的怎么样")


def g(cfg: dict, dotted: str, default=None):
    """按 a.b.c 路径取配置值，缺失返回默认值。"""
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur if cur is not None else default


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ================================================================= 屏幕捕获
class ScreenCapturer:
    """窗口截图：capturekit（独占窗口捕获，被遮挡也能截）优先，失败回退 mss + pygetwindow。
    后端在初始化时探测一次；capturekit 抓帧失败一次即永久回退 mss 并打日志——
    绝不反复报错卡流程（游戏过程中稳定优先）。"""

    def __init__(self, cfg: dict):
        # 采集模式：full = 整屏抓取（默认，屏幕上任何窗口的内容 AI 都能看到）
        #          window = 只抓指定标题关键字的窗口（capturekit 优先，回退 mss）
        self.mode = (g(cfg, "screen.mode", "full") or "full").lower()
        self.keyword = (g(cfg, "screen.window_keyword", "") or "").strip()
        self.backend = None
        self._mss = None
        self._gw = None
        self._mss_lock = threading.Lock()   # mss 非线程安全：处理线程和保底截图线程共用，必须加锁
        self._capture_cls = None
        self._capture_obj = None
        self._window = None
        self._warned = False
        if self.mode == "window":
            self._init_window_backend(g(cfg, "screen.backend", "auto"))
        else:
            self._init_full_backend()

    # ---------------------------------------------------------------- 后端探测
    def _init_full_backend(self):
        """整屏模式：mss 底层就是 DXGI Desktop Duplication，直抓主显示器，稳定省事。
        （capturekit 是专为"独占窗口捕获"设计的，整屏模式用不上它。）"""
        try:
            import mss
            self._mss = mss.mss()
            self.backend = "mss-fullscreen"
            log.info("屏幕捕获: 整屏模式（AI 能看到屏幕上任何内容）")
        except Exception as e:
            self.backend = None
            log.error("屏幕捕获不可用（%s），本轮对话将只发文本", e)

    def _init_window_backend(self, prefer: str):
        # mss 与 pygetwindow 的导入拆开：避免 pygetwindow 出问题时连 mss 也一起不可用
        if prefer in ("auto", "capturekit"):
            try:
                from capturekit import WindowSource  # noqa: F401  # 若你的版本 API 不同，见 README 故障排查
                self._capture_cls = WindowSource
                self.backend = "capturekit"
                log.info("屏幕捕获后端: capturekit（独占窗口，遮挡可截）")
                return
            except Exception as e:
                log.warning("capturekit 不可用（%s），按降级策略改用 mss + pygetwindow", e)
        try:
            import mss
            self._mss = mss.mss()
        except Exception as e:
            log.error("mss 不可用（%s），无法进行窗口截图", e)
        try:
            import pygetwindow as gw
            self._gw = gw
            self.backend = "mss"
            log.info("屏幕捕获后端: mss + pygetwindow（注意：窗口被完全遮挡时可能拿到旧画面）")
        except Exception as e:
            self.backend = None
            log.error("pygetwindow 不可用（%s），窗口模式失效；可改用 screen.mode: full 整屏模式", e)

    # ---------------------------------------------------------------- 窗口定位
    def _find_window(self):
        """按标题关键字找窗口；关键字为空则取当前前台窗口。找不到返回 None（只发文本）。"""
        if self.backend != "mss" or self._gw is None:
            return None
        try:
            active = None
            try:
                active = self._gw.getActiveWindow()
            except Exception:
                pass
            best, best_area = None, 0
            for w in self._gw.getAllWindows():
                try:
                    title = (w.title or "").strip()
                    if not title or w.width < 200 or w.height < 200:
                        continue
                    if self.keyword:
                        hit = self.keyword.lower() in title.lower()
                    else:
                        hit = (w == active)
                    if hit and w.width * w.height > best_area:
                        best, best_area = w, w.width * w.height
                except Exception:
                    continue
            return best
        except Exception as e:
            log.warning("查找游戏窗口失败: %s", e)
            return None

    def _grab_image_mss(self):
        if self._mss is None or self._gw is None:
            return None
        w = self._window
        if w is None or not getattr(w, "title", ""):
            w = self._window = self._find_window()
        if w is None:
            if not self._warned:
                log.warning("未找到匹配窗口（关键字=%r），本轮跳过截图", self.keyword)
                self._warned = True
            return None
        try:
            if getattr(w, "isMinimized", False):
                log.info("窗口已最小化，跳过截图")
                return None
            rect = {"left": int(w.left), "top": int(w.top),
                    "width": int(w.width), "height": int(w.height)}
            if rect["width"] <= 0 or rect["height"] <= 0:
                self._window = None
                return None
            from PIL import Image
            with self._mss_lock:
                shot = self._mss.grab(rect)
            return Image.frombytes("RGB", (shot.width, shot.height), shot.rgb)
        except Exception as e:
            log.warning("mss 截图失败（窗口可能已关闭，将重新定位）: %s", e)
            self._window = None
            return None

    def _grab_image_full(self):
        """整屏抓取：直接 grab 主显示器区域，不做窗口定位。失败返回 None（上层只发文本）。"""
        if self._mss is None:
            return None
        try:
            mon = self._mss.monitors[1]   # monitors[0] 是所有显示器拼接区，[1] 是主显示器
            with self._mss_lock:
                shot = self._mss.grab(mon)
            from PIL import Image
            return Image.frombytes("RGB", (shot.width, shot.height), shot.rgb)
        except Exception as e:
            log.warning("整屏截图失败: %s", e)
            return None

    def _grab_image(self):
        if self.mode == "full":
            return self._grab_image_full()
        if self.backend == "capturekit":
            try:
                # ⚠️ capturekit 的构造参数与抓帧方法名可能随版本变化：
                #    这里按"传窗口标题关键字 → capture() 返回图像"的常见形态调用，
                #    失败一次就回退 mss 并打日志（见 README 故障排查）。绝不静默失败。
                if self._capture_obj is None:
                    self._capture_obj = self._capture_cls(title=self.keyword) if self.keyword \
                        else self._capture_cls()
                frame = self._capture_obj.capture()
                from PIL import Image
                if isinstance(frame, Image.Image):
                    return frame.convert("RGB")
                if isinstance(frame, np.ndarray):
                    return Image.fromarray(frame[:, :, :3]).convert("RGB")
                if isinstance(frame, (bytes, bytearray)):
                    return Image.open(io.BytesIO(bytes(frame))).convert("RGB")
                raise RuntimeError(f"capturekit 返回了未支持的类型: {type(frame)}")
            except Exception as e:
                log.warning("capturekit 抓帧失败（%s），本会话余下时间回退 mss", e)
                self._capture_obj = None
                self.backend = "mss"
        return self._grab_image_mss()

    # ---------------------------------------------------------------- 编码
    def grab_b64(self, max_edge: int = 1024, quality: int = 75) -> Optional[str]:
        """抓 1 帧 → 缩放到最长边 max_edge → JPEG → Base64。失败返回 None（上层只发文本）。"""
        img = self._grab_image()
        if img is None:
            return None
        try:
            w, h = img.size
            scale = max_edge / max(w, h)
            if scale < 1:
                img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=quality)
            import base64
            return base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            log.warning("截图编码失败: %s", e)
            return None

    def close(self):
        try:
            if self._capture_obj is not None and hasattr(self._capture_obj, "close"):
                self._capture_obj.close()
        except Exception:
            pass


# ================================================================= 智能体
def load_agent(name: str, agents_dir: str) -> Optional[dict]:
    path = os.path.join(agents_dir, f"{name}.yaml")
    if not os.path.isfile(path):
        return None
    cfg = load_yaml(path)
    cfg["_id"] = name
    return cfg


def list_agents(agents_dir: str) -> list:
    if not os.path.isdir(agents_dir):
        return []
    return sorted(f[:-5] for f in os.listdir(agents_dir)
                  if f.endswith(".yaml") or f.endswith(".yml"))


# ================================================================= 主程序
class Companion:
    def __init__(self, cfg: dict, agent_name: str, agents_dir: str,
                 api: MimoClient, memory: MemoryStore):
        self.cfg = cfg
        self.agents_dir = agents_dir
        self.api = api
        self.memory = memory
        self.session_id = uuid.uuid4().hex[:12]
        self.running = True

        # ---- 智能体 ----
        self._agent_lock = threading.Lock()
        self.agent = load_agent(agent_name, agents_dir)
        if self.agent is None:
            raise SystemExit(f"智能体 {agent_name!r} 不存在（agents/ 目录内: {list_agents(agents_dir)}）")
        self._pending_greeting: Optional[str] = self.agent.get("greeting")

        # ---- 音频参数 ----
        self.sr = int(g(cfg, "audio.sample_rate", 16000))
        self.frame_ms = int(g(cfg, "audio.frame_ms", 30))
        self.frame_samples = self.sr * self.frame_ms // 1000
        self.start_frames = int(g(cfg, "audio.vad_start_frames", 3))       # 连续 N 帧语音 → 开始说话
        self.silence_ms = int(g(cfg, "audio.silence_ms", 800))             # 静音超时 → 说完
        self.max_record_s = int(g(cfg, "audio.max_record_seconds", 15))    # 录音上限，防噪声死循环
        self.barge_in_frames = int(g(cfg, "audio.barge_in_frames", 6))     # 播报中连续 N 帧语音 → 打断
        self.min_record_s = float(g(cfg, "audio.min_record_seconds", 0.3))

        # VAD 用两个实例：一个在处理线程里判"开始/结束"，一个在音频回调里判"插话"。
        # 为什么分开：webrtcvad 的 C 对象在多线程并发调用下不安全。
        import webrtcvad
        vad_mode = int(g(cfg, "audio.vad_mode", 2))
        self._vad_listen = webrtcvad.Vad(vad_mode)
        self._vad_barge = webrtcvad.Vad(vad_mode)

        self.frame_q: queue.Queue = queue.Queue(maxsize=int(g(cfg, "audio.queue_frames", 256)))
        self._interrupt = threading.Event()          # 插话打断事件（TTS 播放循环检查它）
        self._barge_armed = threading.Event()        # 仅在播报期间置位：开启插话检测
        self._barge_count = 0
        self._interrupt_fired = False

        # ---- 屏幕 ----
        self.capturer = ScreenCapturer(cfg) if g(cfg, "screen.enabled", True) else None
        self.screen_buffer = deque(maxlen=int(g(cfg, "memory.screen_buffer_size", 5)))
        self._last_screen_summary_ts = 0.0
        self._turns_without_screen = 0
        self._idle_frame: Optional[tuple] = None     # (b64, ts) 保底帧
        self._last_user_speech_ts = 0.0

        # ---- 后台任务 ----
        self.jobs: queue.Queue = queue.Queue()
        self._job_flags: dict = {}
        self._job_lock = threading.Lock()

        # ---- 会话状态展示 ----
        self.state = "空闲"
        self._state_lock = threading.Lock()

    # ------------------------------------------------------------ 状态
    def _set_state(self, s: str):
        with self._state_lock:
            self.state = s

    # ------------------------------------------------------------ 音频输入
    def _audio_callback(self, indata, frames, time_info, status):
        """PortAudio 回调：只做"搬帧入队 + 插话检测"，绝不在这里做网络/重活，
        否则会丢音频块导致识别断续。"""
        if status:
            log.debug("音频流状态: %s", status)
        data = bytes(indata)
        try:
            self.frame_q.put_nowait(data)
        except queue.Full:
            try:   # 队列满：丢最旧的帧保实时性
                self.frame_q.get_nowait()
                self.frame_q.put_nowait(data)
            except queue.Empty:
                pass
        # 插话检测：仅播报期间武装。连续 N 帧语音 → 立即打断 TTS。
        if self._barge_armed.is_set() and not self._interrupt_fired:
            try:
                speech = self._vad_barge.is_speech(data, self.sr)
            except Exception:
                return
            if speech:
                self._barge_count += 1
                if self._barge_count >= self.barge_in_frames:
                    self._interrupt_fired = True
                    self._interrupt.set()          # 播放循环看到它就丢弃剩余音频
                    log.info("检测到用户插话，立即打断当前播报")
            else:
                self._barge_count = 0

    # ------------------------------------------------------------ VAD 状态机
    def _wait_for_speech(self) -> bool:
        """[空闲] 等待用户开口：连续 start_frames 帧语音才算真的在说话（滤掉环境噪声）。"""
        count = 0
        first = None
        while self.running:
            try:
                frame = self.frame_q.get(timeout=0.2)
            except queue.Empty:
                count = 0
                continue
            try:
                speech = self._vad_listen.is_speech(frame, self.sr)
            except Exception:
                continue
            if speech:
                count += 1
                if count >= self.start_frames:
                    self._first_frame = first or frame
                    return True
                elif first is None:
                    first = frame
            else:
                count = 0
                first = None
        return False

    def _record_utterance(self) -> Optional[bytes]:
        """[录音中] 收集 PCM，直到静音超过 silence_ms 或达到 max_record_s。"""
        buf = [self._first_frame]
        started = time.time()
        last_speech = started
        while self.running:
            try:
                frame = self.frame_q.get(timeout=0.3)
            except queue.Empty:
                if time.time() - started >= self.max_record_s:
                    break
                continue
            buf.append(frame)
            try:
                if self._vad_listen.is_speech(frame, self.sr):
                    last_speech = time.time()
            except Exception:
                pass
            if time.time() - last_speech >= self.silence_ms / 1000.0:
                break                      # 静音超时 → 认为说完了
            if time.time() - started >= self.max_record_s:
                log.info("录音达到最长 %ds 上限，自动截断", self.max_record_s)
                break
        return b"".join(buf)

    # ------------------------------------------------------------ 一轮对话
    def _pick_image(self) -> Optional[str]:
        """本轮截图：优先现抓 1 帧；抓不到就用保底帧（若还新鲜）。返回 base64 或 None。"""
        if self.capturer is None or not self.api.vision_ok:
            return None
        max_edge = int(g(self.cfg, "screen.max_edge", 1024))
        quality = int(g(self.cfg, "screen.jpeg_quality", 75))
        b64 = self.capturer.grab_b64(max_edge, quality)
        if b64 is None and self._idle_frame:
            idle_b64, idle_ts = self._idle_frame
            if time.time() - idle_ts < int(g(self.cfg, "screen.interval_seconds", 25)) * 3:
                b64 = idle_b64
        return b64

    def _build_system_prompt(self) -> str:
        """system = 智能体人设 + 长期记忆摘要 + 近期屏幕记忆（自动拼接，不用用户手写）。"""
        parts = [(self.agent.get("system_prompt") or "").strip()]
        lt = self.memory.latest_long_term(self.agent["_id"],
                                          shared=bool(self.agent.get("share_memory")))
        if lt:
            parts.append("【长期记忆摘要（自动注入）】\n" + lt)
        screens = self.memory.recent_screen_summaries(
            self.agent["_id"],
            max_age_minutes=float(g(self.cfg, "memory.screen_summary_max_age_minutes", 30)),
            limit=2)
        if screens:
            body = "\n".join(f"- [{fmt_ts(t)}] {c}" for c, t in screens)
            parts.append("【近期屏幕记忆（自动注入）】\n" + body)
        return "\n\n".join(p for p in parts if p)

    def _chat(self, text: str, img_b64: Optional[str]) -> Optional[str]:
        """构造 messages 并请求 flash。
        顺序：system(人设+长期记忆+屏幕记忆) → 最近 N 条历史 → 当前 user(文本+当轮截图)。
        图像只放在当轮 user 消息里，永不进入历史与数据库。"""
        agent_id = self.agent["_id"]
        history = self.memory.recent_messages(agent_id, int(g(self.cfg, "memory.history_window", 20)))
        if img_b64:
            content = [
                {"type": "text", "text": text + "\n（附：我当前游戏画面的截图）"},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ]
        else:
            content = text
        messages = ([{"role": "system", "content": self._build_system_prompt()}]
                    + history + [{"role": "user", "content": content}])
        try:
            return self.api.chat_reply(messages, has_image=bool(img_b64))
        except MimoApiError as e:
            log.error("对话请求失败: %s", e)
            return None          # 上层用固定兜底话术

    def _speak(self, text: str):
        """TTS 播报（低延迟流式优先）+ 打断支持 + 两级降级。
        流式失败 → 重试一次非流式 wav → 仍失败只在控制台打印文本。"""
        if not text:
            return
        voice = self.agent.get("tts_voice") or g(self.cfg, "tts.voice", "mimo_default")
        style = g(self.cfg, "tts.style_hint", "用自然、陪伴感的语气说")
        sr = int(g(self.cfg, "tts.sample_rate", 24000))   # ⚠️ 不确定项：按实际 API 返回调整

        self._interrupt.clear()
        self._barge_count = 0
        self._interrupt_fired = False
        self._barge_armed.set()      # 武装插话检测（在音频回调里生效）
        gen = None
        ok = False
        try:
            try:   # 路径 1：流式 pcm16，收到第一块就开播
                gen = self.api.tts_stream(text, voice, style)
                played = False
                with sd.OutputStream(samplerate=sr, dtype="int16", channels=1) as out:
                    for pcm in gen:        # 流式音频逐块写入播放（这就是"拼接"）
                        if self._interrupt.is_set():
                            break          # 插话：丢弃剩余音频，不补播
                        if pcm:
                            out.write(np.frombuffer(pcm, dtype="<i2"))
                            played = True
                if self._interrupt.is_set():
                    log.info("播报被打断，剩余音频已丢弃")
                    ok = True
                elif not played:
                    raise RuntimeError("整个流没有收到任何音频块")
                else:
                    ok = True
            except Exception as e:
                if self._interrupt.is_set():
                    ok = True              # 打断不算失败，不触发回退
                else:
                    log.warning("TTS 流式失败（%s），自动重试一次非流式 wav", e)

            if not ok:   # 路径 2：非流式 wav 回退
                try:
                    wav = self.api.tts_once(text, voice, style)
                    pcm, rate, _ch = wav_to_pcm16(wav)
                    with sd.OutputStream(samplerate=rate, dtype="int16", channels=1) as out:
                        out.write(pcm)
                    ok = True
                except Exception as e:
                    log.error("TTS 非流式也失败（%s），本轮只打印文本不发声", e)

            if not ok:   # 路径 3：只打印
                print(f"  【{self.agent.get('name')}·未播出】{text}")
        finally:
            self._barge_armed.clear()
            self._barge_count = 0
            if gen is not None:
                try:
                    gen.close()            # 关掉流式响应，别让 httpx 继续下载
                except Exception:
                    pass

    def _beep(self):
        """ASR 失败等场景的短提示音（可配置关闭）。"""
        if not g(self.cfg, "audio.beep_on_error", True):
            return
        try:
            sr = 24000
            t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False)
            tone = (np.sin(2 * np.pi * 660 * t) * 0.15 * 32767).astype("<i2")
            with sd.OutputStream(samplerate=sr, dtype="int16", channels=1) as out:
                out.write(tone)
        except Exception:
            pass

    def _process_turn(self, pcm: bytes):
        agent_id = self.agent["_id"]
        # ---- ASR ----
        self._set_state("识别中")
        try:
            text = self.api.transcribe(pcm16_to_wav_bytes(pcm, self.sr),
                                       language=str(self.agent.get("asr_language", "zh")))
        except MimoApiError as e:
            log.error("ASR 失败（%s），回到空闲", e)
            self._beep()
            return
        text = (text or "").strip()
        if not text:
            log.info("识别结果为空，忽略本轮")
            return
        self._last_user_speech_ts = time.time()
        print(f"  [你说] {text}")

        # 用户主动要求总结 → 立刻排一个屏幕总结任务
        if any(k in text for k in SUMMARY_KEYWORDS) and self.screen_buffer:
            self._enqueue_job("screen_summary")

        # ---- 截图 + 对话 ----
        self._set_state("思考中")
        img = self._pick_image()
        if img:
            self._turns_without_screen = 0
        else:
            self._turns_without_screen += 1
        reply = self._chat(text, img)
        if not reply:
            reply = FALLBACK_REPLY    # flash 挂了：固定话术兜底，保证语音闭环不断

        # ---- 播报 ----
        self._set_state("播报中")
        self._speak(reply)
        self._set_state("空闲")
        print(f"  [{self.agent.get('name')}] {reply}")

        # ---- 记忆写入（文本入库；图像永不入库）----
        self.memory.add_message(agent_id, "user", text, self.session_id)
        self.memory.add_message(agent_id, "assistant", reply, self.session_id)

        # ---- 屏幕观察缓冲：本轮截图 + AI 对该画面的回应（文本形式，不是图片）----
        if img and self.agent.get("screen_summary_enabled", True):
            self.screen_buffer.append((img, reply[:60], time.time()))

        # ---- 触发检查：屏幕阶段总结 & 长期记忆压缩 ----
        self._maybe_trigger_summaries()

    # ------------------------------------------------------------ 后台任务
    def _enqueue_job(self, kind: str):
        with self._job_lock:
            if self._job_flags.get(kind):
                return            # 同类任务已在排队/执行，去重，避免刷屏
            self._job_flags[kind] = True
        self.jobs.put(kind)

    def _maybe_trigger_summaries(self):
        """记忆摘要触发条件（都在这里判断，实际执行放后台线程，不阻塞语音）。"""
        agent_id = self.agent["_id"]
        if self.agent.get("memory_enabled", True):
            if self.memory.count_unarchived(agent_id) > int(
                    g(self.cfg, "memory.summarize_threshold", 200)):
                self._enqueue_job("summarize_history")
        if self.screen_buffer:
            interval = float(g(self.cfg, "memory.screen_summary_interval_minutes", 3.5)) * 60
            full = len(self.screen_buffer) >= self.screen_buffer.maxlen
            if full or (time.time() - self._last_screen_summary_ts) >= interval:
                self._enqueue_job("screen_summary")

    def _do_summarize_history(self):
        """长期记忆压缩：把较早的一批消息交给 flash 生成 200 字内摘要，旧消息标 archived=1。"""
        agent_id = self.agent["_id"]
        limit = int(g(self.cfg, "memory.summarize_batch", 100))
        rows = self.memory.pending_summary_rows(agent_id, limit)
        if len(rows) < 2:
            return
        block = "\n".join(f"{r}: {c}" for _i, r, c in rows)
        prompt = ("以下是此前的对话记录，请提取用户偏好、称呼、常玩游戏、重要事件、情绪倾向，"
                  "用200字以内中文要点式总结，供后续对话参考。只输出总结本身。\n\n" + block)
        try:
            summary = self.api.chat_text(prompt)
        except MimoApiError as e:
            log.error("长期摘要生成失败（不阻塞语音）: %s", e)
            return
        if summary:
            self.memory.add_summary(agent_id, "long_term", summary)
            self.memory.mark_archived([i for i, _r, _c in rows])

    def _do_screen_summary(self):
        """屏幕阶段总结：把缓冲里的多张截图降采样后批量发 flash，要 100 字内进度总结。
        屏幕图只在这里和当轮请求里使用，绝不入库。"""
        if not self.screen_buffer:
            return
        agent_id = self.agent["_id"]
        items = list(self.screen_buffer)
        self.screen_buffer.clear()
        if not self.api.vision_ok:
            log.warning("图像输入不可用，跳过屏幕总结（丢弃 %d 张缓冲截图）", len(items))
            return
        max_edge = int(g(self.cfg, "screen.summary_max_edge", 768))
        quality = int(g(self.cfg, "screen.summary_jpeg_quality", 60))
        parts = [{
            "type": "text",
            "text": ("根据这几张游戏截图，用100字以内中文总结当前游戏进度、玩家状态、"
                     "值得记住的信息。不要逐帧描述。只输出总结。"),
        }]
        for b64, _txt, _ts in items:
            parts.append({"type": "text", "text": "（下一张截图）"})
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        try:
            summary = self.api.chat_reply([{"role": "user", "content": parts}], has_image=True)
        except MimoApiError as e:
            log.error("屏幕总结生成失败（不阻塞语音）: %s", e)
            return
        if summary:
            self.memory.add_summary(agent_id, "screen", summary)
        self._last_screen_summary_ts = time.time()

    def _worker(self):
        """后台单线程任务队列：记忆压缩、屏幕总结都在这里跑，
        用单线程 + 队列是为了避免多线程并发写 SQLite，也保证绝不阻塞语音主循环。"""
        while True:
            kind = self.jobs.get()
            if kind is None:
                break
            try:
                if kind == "summarize_history":
                    self._do_summarize_history()
                elif kind == "screen_summary":
                    self._do_screen_summary()
            except Exception as e:
                log.error("后台任务 %s 异常: %s", kind, e)
            finally:
                with self._job_lock:
                    self._job_flags.pop(kind, None)

    # ------------------------------------------------------------ 保底截图
    def _periodic_capture_loop(self):
        """用户长时间不说话时，每 interval_seconds 抓 1 帧存为"保底帧"；
        只有最近几轮都没带屏幕信息时才真正使用（省 token）。"""
        interval = float(g(self.cfg, "screen.interval_seconds", 25))
        while self.running:
            time.sleep(interval)
            if self.capturer is None or not self.api.vision_ok:
                continue
            if time.time() - self._last_user_speech_ts < interval:
                continue          # 用户刚说过话，本轮已有新鲜截图，不需要保底
            if self._turns_without_screen < 2:
                continue          # 最近对话已经带过屏幕信息，别浪费 token
            b64 = self.capturer.grab_b64(int(g(self.cfg, "screen.max_edge", 1024)),
                                         int(g(self.cfg, "screen.jpeg_quality", 75)))
            if b64:
                self._idle_frame = (b64, time.time())
                log.debug("已更新保底截图帧")

    # ------------------------------------------------------------ 智能体切换
    def switch_agent(self, name: str):
        """切换智能体：换 system prompt 和音色；持久化记忆按 agent_name 隔离，切换不丢。"""
        cfg = load_agent(name, self.agents_dir)
        if cfg is None:
            print(f"智能体 {name!r} 不存在。可用: {list_agents(self.agents_dir)}")
            return
        with self._agent_lock:
            self.agent = cfg
            self._pending_greeting = cfg.get("greeting")
        self._interrupt.set()     # 若正在播报旧智能体，打断它
        print(f"已切换智能体：{cfg.get('name')}（记忆保留）")

    def _cycle_agent(self):
        agents = list_agents(self.agents_dir)
        if len(agents) < 2:
            return
        cur = self.agent.get("_id")
        nxt = agents[(agents.index(cur) + 1) % len(agents)] if cur in agents else agents[0]
        self.switch_agent(nxt)

    # ------------------------------------------------------------ 控制台命令
    def _console_loop(self):
        try:
            for line in sys.stdin:
                if not self.running:
                    break
                cmd = line.strip()
                if not cmd:
                    continue
                if cmd.startswith("agent "):
                    self.switch_agent(cmd.split(None, 1)[1].strip())
                elif cmd == "agents":
                    print("可用智能体:", ", ".join(list_agents(self.agents_dir)))
                elif cmd == "status":
                    print(f"状态={self.state} 智能体={self.agent.get('name')} "
                          f"视觉={'开' if self.api.vision_ok else '降级(纯文本)'} "
                          f"截图后端={self.capturer.backend if self.capturer else '关闭'}")
                elif cmd in ("quit", "exit"):
                    self.stop()
                else:
                    print("可用命令: agent <名字> | agents | status | quit")
        except Exception:
            pass

    # ------------------------------------------------------------ 主循环
    def _loop(self):
        while self.running:
            # 开场白（或切换智能体后的问候）：等空闲时说
            if self._pending_greeting:
                greeting, self._pending_greeting = self._pending_greeting, None
                self._set_state("播报中")
                self._speak(greeting)
                self._set_state("空闲")
            self._set_state("空闲")
            if not self._wait_for_speech():
                break
            self._set_state("录音中")
            pcm = self._record_utterance()
            if not pcm or len(pcm) < int(self.sr * 2 * self.min_record_s):
                continue        # 太短，视为误触发
            try:
                self._process_turn(pcm)
            except Exception as e:
                # 兜底：任何未预料异常都不能让主循环退出（游戏陪伴要"永远在线"）
                log.exception("处理对话轮次异常: %s", e)
                self._set_state("空闲")

    # ------------------------------------------------------------ 生命周期
    def start(self):
        self._set_state("启动中")
        self.stream = sd.InputStream(samplerate=self.sr, channels=1, dtype="int16",
                                     blocksize=self.frame_samples, callback=self._audio_callback)
        self.stream.start()
        self._worker_t = threading.Thread(target=self._worker, daemon=True)
        self._worker_t.start()
        self._periodic_t = threading.Thread(target=self._periodic_capture_loop, daemon=True)
        self._periodic_t.start()
        self._console_t = threading.Thread(target=self._console_loop, daemon=True)
        self._console_t.start()

        # F8 循环切换智能体（keyboard 库是全局钩子，个别环境需要管理员权限；失败不影响主流程）
        try:
            import keyboard
            keyboard.on_press_key(g(self.cfg, "agent.hotkey", "f8"),
                                  lambda _e: threading.Thread(
                                      target=self._cycle_agent, daemon=True).start())
            log.info("已注册热键 F8 = 切换下一个智能体")
        except Exception as e:
            log.warning("全局热键不可用（%s），请用控制台命令 agent <名字> 切换", e)

        print("=" * 52)
        print(f" 就绪 | 智能体: {self.agent.get('name')} | 说句话开始吧")
        print(" 命令: agent <名字> / agents / status / quit   热键: F8")
        print("=" * 52)
        self._loop()

    def stop(self):
        if not self.running:
            return
        self.running = False
        self._interrupt.set()
        try:
            self.jobs.put(None)
        except Exception:
            pass
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        if self.capturer:
            self.capturer.close()
        self.memory.close()
        print("\n已退出，资源已释放。")


# ================================================================= 入口
def main():
    parser = argparse.ArgumentParser(description="小米 MIMO 实时语音游戏陪玩")
    parser.add_argument("--agent", default=None, help="智能体名（agents/ 下的文件名）")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--list-agents", action="store_true", help="列出所有智能体后退出")
    parser.add_argument("--reset-memory", action="store_true", help="清空记忆库（需二次确认）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv()
    agents_dir = "agents"

    if args.list_agents:
        print("可用智能体:", ", ".join(list_agents(agents_dir)) or "（无）")
        return

    # 配置文件：优先 config.yaml，缺失则退回 config.example.yaml 并提醒
    cfg_path = args.config if os.path.isfile(args.config) else "config.example.yaml"
    if not os.path.isfile(args.config):
        print(f"[提示] 未找到 {args.config}，已使用 {cfg_path}（正式使用请复制 config.example.yaml → config.yaml）")
    cfg = load_yaml(cfg_path)

    # API Key 检查：不硬编码，只从 .env / 环境变量读取
    api_key = os.getenv("MIMO_API_KEY", "").strip()
    if not api_key:
        print("错误：未找到 MIMO_API_KEY。\n"
              "  请复制 .env.example 为 .env，在其中填入 MIMO_API_KEY=你的密钥，"
              "或先设置环境变量后重试。")
        sys.exit(1)

    memory = MemoryStore(g(cfg, "memory.db_path", "data/memory.db"))

    if args.reset_memory:
        ans = input("确认清空所有智能体的对话与摘要记忆？此操作不可恢复 [yes/no]: ").strip().lower()
        if ans == "yes":
            memory.reset()
            print("记忆库已清空。")
        else:
            print("已取消。")
        memory.close()
        return

    api = MimoClient(
        api_key=api_key,
        base_url=g(cfg, "api.base_url", "https://api.xiaomimimo.com/v1"),
        chat_model=g(cfg, "models.chat", "mimo-v2.6-flash"),
        asr_model=g(cfg, "models.asr", "mimo-v2.5-asr"),
        tts_model=g(cfg, "models.tts", "mimo-v2.5-tts"),
        connect_timeout=float(g(cfg, "api.connect_timeout", 5)),
        read_timeout=float(g(cfg, "api.read_timeout", 30)),
        tts_sample_rate=int(g(cfg, "tts.sample_rate", 24000)),
        vision_enabled=bool(g(cfg, "vision.enabled", True)),
        clone_model=g(cfg, "tts.clone_model", "mimo-v2.5-tts-voiceclone"),
        voice_ref_path=str(g(cfg, "tts.voice_ref", "") or ""),
    )

    agent_name = args.agent or g(cfg, "agent.default", "example")
    app = Companion(cfg, agent_name, agents_dir, api, memory)

    def _sigint(_s, _f):
        print("\n收到退出信号，正在优雅退出...")
        app.stop()

    signal.signal(signal.SIGINT, _sigint)
    try:
        app.start()
    except KeyboardInterrupt:
        app.stop()


if __name__ == "__main__":
    main()
