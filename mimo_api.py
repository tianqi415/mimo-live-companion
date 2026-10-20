# -*- coding: utf-8 -*-
"""
mimo_api.py —— 小米 MIMO API 三接口统一封装（对话/视觉 flash、ASR、TTS）
======================================================================
核心约定（端点容易写错，务必注意）：
  * 三个模型全部走 OpenAI 兼容的 POST /chat/completions
      - ASR 不是 /audio/transcriptions
      - TTS 不是 /audio/speech
  * 认证统一用请求头 api-key（openai SDK 的 api_key 参数会自动放进该头）
  * 所有不确定的响应结构都做"多路径提取 + 明确报错"，绝不会猜错后静默失败
"""
from __future__ import annotations

import base64
import io
import logging
import os
import wave
from typing import Iterator, List, Optional

log = logging.getLogger("mimo.api")


class MimoApiError(RuntimeError):
    """API 层统一错误：网络、鉴权、响应结构异常都收敛到这里，方便上层做降级。"""


def pcm16_to_wav_bytes(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """把裸 PCM16 数据包成 WAV（加 RIFF 头）。
    为什么需要：ASR 只收 wav/mp3；把录音缓冲发给 mimo-v2.5-asr 前必须先包一层 WAV 头。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)          # 16-bit = 2 字节
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def wav_to_pcm16(wav_bytes: bytes):
    """WAV → (numpy int16 数组, 采样率, 声道数)。TTS 非流式回退拿到的 wav 用它转成可播放 PCM。"""
    import numpy as np
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        data = w.readframes(w.getnframes())
    if width != 2:
        raise MimoApiError(f"TTS 返回的 wav 位深不是 16-bit（{width*8}bit），无法直接播放")
    return np.frombuffer(data, dtype="<i2"), rate, ch


def _strip_images(messages: List[dict]) -> List[dict]:
    """把消息里的图像 part 全部删掉——图像输入不被支持时的纯文本降级路径。"""
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            texts = [p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"]
            out.append({**m, "content": "\n".join(t for t in texts if t)})
        else:
            out.append(m)
    return out


def _safe_dump(obj) -> str:
    try:
        return obj.model_dump_json() if hasattr(obj, "model_dump_json") else str(obj)
    except Exception:
        return repr(obj)


def _extract_audio(resp) -> Optional[bytes]:
    """从非流式 TTS 响应里提取音频字节（多路径）。
    ⚠️ 非流式响应中音频字段的确切位置公开文档不明确：
       这里依次尝试 message.audio.data / message.content(base64 或 data URL) / content parts。
       全部失败返回 None → 上层明确报错，请把控制台日志发回来，我按实际响应结构调整这里。
       另外：如果解出来是裸 pcm（没有 RIFF 头）也返回 None——采样率不明，不硬猜。"""
    try:
        msg = resp.choices[0].message
    except Exception:
        return None
    raw = None
    a = getattr(msg, "audio", None)
    if a is not None:
        raw = getattr(a, "data", None) or (a if isinstance(a, str) else None)
    if not raw:
        c = getattr(msg, "content", None)
        if isinstance(c, str) and c.strip():
            s = c.strip()
            raw = s.split(",", 1)[1] if s.startswith("data:") else s
        elif isinstance(c, list):
            for part in c:
                for attr in ("audio", "data"):
                    v = getattr(part, attr, None)
                    if isinstance(v, dict):
                        v = v.get("data")
                    if isinstance(v, str) and v:
                        raw = v
                        break
                if raw:
                    break
    if not raw:
        return None
    try:
        data = base64.b64decode(raw)
    except Exception:
        return None
    return data if data[:4] == b"RIFF" else None


class MimoClient:
    def __init__(self, api_key: str, base_url: str,
                 chat_model: str, asr_model: str, tts_model: str,
                 connect_timeout: float = 5, read_timeout: float = 30,
                 tts_sample_rate: int = 24000, vision_enabled: bool = True,
                 max_retries: int = 1,
                 clone_model: str = "mimo-v2.5-tts-voiceclone",
                 voice_ref_path: str = ""):
        try:
            import httpx
            from openai import OpenAI
        except ImportError as e:
            raise MimoApiError(f"缺少依赖（{e}），请先执行: pip install -r requirements.txt")

        self.chat_model = chat_model
        self.asr_model = asr_model
        self.tts_model = tts_model
        self.tts_sample_rate = tts_sample_rate   # ⚠️ 流式 PCM16 采样率，若播放音调/语速不对请改 config.yaml

        # 克隆音色：提供参考音频路径时自动切换到 voiceclone 模型（音频以 data URL 随每次请求传入）。
        # 为什么这样设计：voiceclone 没有"注册音色ID"的概念，参考音频必须每次随请求带上。
        self.clone_model = clone_model
        self._voice_ref_data_url = None
        if voice_ref_path:
            if os.path.isfile(voice_ref_path):
                with open(voice_ref_path, "rb") as f:
                    ref_raw = f.read()
                ref_mime = "audio/wav" if voice_ref_path.lower().endswith(".wav") else "audio/mpeg"
                self._voice_ref_data_url = "data:%s;base64,%s" % (
                    ref_mime, base64.b64encode(ref_raw).decode())
                log.info("已加载克隆音色参考音频 %s（%.2fMB），TTS 模型切换为 %s",
                         voice_ref_path, len(ref_raw) / 1e6, clone_model)
            else:
                log.warning("克隆音色参考音频不存在（%s），回退预置音色", voice_ref_path)

        # 实时场景超时策略：连接必须短（5s 快速失败），读取 30s；SDK 自带重试压到很小，
        # 避免网络抖动时自动重试叠加出不可控延迟。
        self._timeout = httpx.Timeout(connect=connect_timeout, read=read_timeout,
                                      write=connect_timeout, pool=connect_timeout)
        self.client = OpenAI(api_key=api_key, base_url=base_url,
                             timeout=self._timeout, max_retries=max_retries)

        self.vision_enabled = vision_enabled
        self._vision_fail_strikes = 0   # 带图失败计数：累计 3 次判定"该环境图像输入不可用"，本会话降级纯文本

    # ---------------------------------------------------------------- 对话/视觉
    def _chat_raw(self, messages: List[dict], stream: bool = False):
        try:
            return self.client.chat.completions.create(model=self.chat_model,
                                                       messages=messages, stream=stream)
        except Exception as e:
            raise MimoApiError(f"chat/completions 调用失败: {e}")

    @property
    def vision_ok(self) -> bool:
        return self.vision_enabled and self._vision_fail_strikes < 3

    def chat_reply(self, messages: List[dict], has_image: bool = False) -> str:
        """对话主接口。带图失败时按降级策略改成纯文本重发一次，绝不阻塞语音主流程。"""
        try:
            resp = self._chat_raw(messages)
            return (resp.choices[0].message.content or "").strip()
        except MimoApiError as e:
            if has_image and self.vision_enabled:
                self._vision_fail_strikes += 1
                log.warning("带图请求失败(%s)，降级为纯文本重发一次%s", e,
                            "；连续失败3次，本会话不再附图" if self._vision_fail_strikes >= 3 else "")
                try:
                    resp = self._chat_raw(_strip_images(messages))
                    return (resp.choices[0].message.content or "").strip()
                except MimoApiError as e2:
                    raise MimoApiError(f"纯文本重发仍失败: {e2}") from e2
            raise

    def chat_text(self, prompt: str) -> str:
        """纯文本单轮请求（长期记忆压缩、屏幕阶段总结用），永远不带图像。"""
        resp = self._chat_raw([{"role": "user", "content": prompt}])
        return (resp.choices[0].message.content or "").strip()

    # ---------------------------------------------------------------- ASR
    def transcribe(self, wav_bytes: bytes, language: str = "zh") -> str:
        """mimo-v2.5-asr。注意：走 chat/completions，音频 base64 放 input_audio content part。
        asr_options 是请求体顶层字段（不是 query 参数），用 extra_body 透传。"""
        b64 = base64.b64encode(wav_bytes).decode()
        try:
            resp = self.client.chat.completions.create(
                model=self.asr_model,
                messages=[{
                    "role": "user",
                    "content": [{
                        "type": "input_audio",
                        "input_audio": {"data": f"data:audio/wav;base64,{b64}"},
                    }],
                }],
                extra_body={"asr_options": {"language": language}},
            )
        except Exception as e:
            raise MimoApiError(f"ASR 调用失败: {e}")
        try:
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            raise MimoApiError(f"ASR 响应结构异常: {e}")

    # ---------------------------------------------------------------- TTS
    def _tts_target(self, voice: str, fmt: str):
        """选择 TTS 的目标模型和 audio 参数：
        启用了克隆参考音频 → voiceclone 模型 + data URL 形式的 voice；
        未启用 → 预置音色模型 + 音色名。"""
        if self._voice_ref_data_url:
            return self.clone_model, {"format": fmt, "voice": self._voice_ref_data_url}
        return self.tts_model, {"format": fmt, "voice": voice or "mimo_default"}

    def tts_stream(self, text: str, voice: str, style_hint: str = "") -> Iterator[bytes]:
        """流式合成：stream=True + audio.format=pcm16，逐块 yield 裸 PCM16 字节。
        上层收到第一块就开播 → 低首字延迟。必须有 role:"assistant" 的消息承载待合成文本。"""
        messages = []
        if style_hint:   # role:"user" 提供语气/风格提示（可选）
            messages.append({"role": "user", "content": style_hint})
        messages.append({"role": "assistant", "content": text})

        model, audio_param = self._tts_target(voice, "pcm16")
        try:
            stream = self.client.chat.completions.create(
                model=model, messages=messages, stream=True,
                extra_body={"audio": audio_param},
            )
        except Exception as e:
            raise MimoApiError(f"TTS 流式请求失败: {e}")

        first = True
        for chunk in stream:
            data = None
            try:
                # ⚠️ 官方文档示例里 delta.audio 是 dict（{"data": ...}），也可能是对象；两种都兼容
                a = getattr(chunk.choices[0].delta, "audio", None)
                if isinstance(a, dict):
                    data = a.get("data")
                else:
                    data = getattr(a, "data", None)
            except (AttributeError, IndexError):
                data = None
            if first:
                # ⚠️ 首块把原始结构打进日志：如果线上字段名与预期不同，按日志调整本函数
                log.debug("TTS 首块原始结构: %s", _safe_dump(chunk)[:500])
                first = False
            if data:
                try:
                    yield base64.b64decode(data)
                except Exception as e:
                    log.warning("TTS 音频块 base64 解码失败（忽略该块）: %s", e)
        # 说明：若整个流一个音频块都没有，上层会因"没播到声音"而回退到非流式

    def tts_once(self, text: str, voice: str, style_hint: str = "") -> bytes:
        """非流式回退：stream=False + audio.format=wav，一次性取回 WAV 字节。
        ⚠️ 音频字段位置做了多路径提取；失败会明确抛错（见 _extract_audio 注释）。"""
        messages = []
        if style_hint:
            messages.append({"role": "user", "content": style_hint})
        messages.append({"role": "assistant", "content": text})
        model, audio_param = self._tts_target(voice, "wav")
        try:
            resp = self.client.chat.completions.create(
                model=model, messages=messages, stream=False,
                extra_body={"audio": audio_param},
            )
        except Exception as e:
            raise MimoApiError(f"TTS 非流式请求失败: {e}")
        audio = _extract_audio(resp)
        if not audio:
            raise MimoApiError("TTS 非流式响应中未识别出音频字段（见 README 故障排查，把日志发我调整）")
        return audio
