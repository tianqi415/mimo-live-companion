# MIMO Live Companion · 实时语音游戏陪玩

一个跑在 Windows 上的开源实时语音陪伴助手：和你**语音聊天**、"看得见"你的**游戏画面**、拥有**永久记忆**、可自由定义**智能体人格与音色**。底层全部调用小米 MIMO 开放平台 API（OpenAI 兼容接口）。

## 特性

- **实时语音对话**：麦克风 VAD 自动断句，说完即识别、即回复、即播报；播报中插话立即打断
- **屏幕感知**：默认整屏捕获（一行配置改为只截指定窗口），每轮对话自动附带最新画面，AI 结合画面回应
- **永久记忆**：SQLite 本地存储全部对话（WAL 模式）；消息攒多后后台自动压缩成长期摘要；游戏画面每几分钟自动总结为"近期屏幕记忆"
- **多智能体**：`agents/` 目录下一个 YAML 一个人格一个音色，运行中热键/命令随时切换，记忆按智能体隔离
- **音色克隆（可选）**：提供一段人声样本（mp3/wav），智能体即可用你指定的声音说话
- **全链路降级**：ASR / 截图 / 对话 / TTS 任一环节失败都不会卡死主流程，保证游戏过程"永远在线"

## 项目结构

```
mimo_live_companion.py   主脚本（状态机、音频/VAD、屏幕捕获、TTS 播放与打断、智能体切换）
memory.py                SQLite 记忆层（消息、长期摘要、屏幕阶段总结）
mimo_api.py              MIMO 三接口封装（对话/视觉、ASR、TTS）与降级路径
config.example.yaml      全部参数（复制为 config.yaml 使用，含中文注释）
.env.example             API Key 模板（复制为 .env 使用）
agents/                  智能体配置目录（example.yaml 为模板，复制后自建）
assets/                  自备音色克隆参考音频的存放目录（可选）
```

## 快速开始

环境：**Windows 10/11 + Python 3.10+**（建议戴耳机，避免外放声音触发误打断）

```bat
:: 1. 安装依赖
pip install -r requirements.txt

:: 2. 配置密钥：复制 .env.example 为 .env，填入你的小米 MIMO API Key
copy .env.example .env
notepad .env

:: 3. 生成配置：复制 config.example.yaml 为 config.yaml
copy config.example.yaml config.yaml

:: 4. 创建你的智能体（见下一节），然后运行
python mimo_live_companion.py --agent example
```

运行中可用命令：`agent <名字>` 切换智能体 / `agents` 列出全部 / `status` 查看状态 / `quit` 退出；按 **F8** 循环切换智能体。

## 创建你的智能体

复制 `agents/example.yaml` 为 `agents/你的名字.yaml`，按注释填写：

| 字段 | 说明 |
|---|---|
| `name` | 显示名 |
| `system_prompt` | 人设与说话方式（自动拼接长期记忆与屏幕记忆，无需手写） |
| `tts_voice` | 预置音色（冰糖/茉莉/苏打/白桦等），或配合 `voice_ref` 使用克隆音色 |
| `asr_language` | 识别语言，默认 `zh` |
| `reply_max_sentences` | 回复长度上限（句） |
| `screen_summary_enabled` | 是否启用屏幕阶段性总结 |
| `memory_enabled` | 是否启用长期记忆压缩 |
| `share_memory` | true = 与其他智能体共享长期记忆摘要 |
| `greeting` | 启动开场白（可选） |

## 记忆系统

- 位置：`data/memory.db`（SQLite + WAL；**屏幕图像永不入库**）
- 每轮对话文字自动入库；未归档消息超过 200 条时，后台自动压缩为 200 字内长期摘要
- 游戏画面每 3~5 分钟自动总结一次，作为"近期屏幕记忆"注入后续对话（超过 30 分钟不再注入）
- 清空记忆：`python mimo_live_companion.py --reset-memory`（输入 yes 二次确认）

## 音色克隆（可选）

把一段干净的人声样本（mp3/wav，≤10MB，建议 10~60 秒）放进 `assets/`，然后在 `config.yaml`：

```yaml
tts:
  voice_ref: "assets/你的样本.mp3"
```

重启即生效；留空 `""` 则使用预置音色。

> ⚠️ 实测 `mimo-v2.5-tts-voiceclone` 合成延迟波动较大（6 秒音频最好约 39 秒、常见 3~4 分钟，服务端排队所致），实时对话场景建议使用预置音色。

## 常见问题

| 现象 | 处理 |
|---|---|
| 启动提示未找到 MIMO_API_KEY | `.env` 未配置或不在脚本同目录 |
| 声音变快/变慢/发尖 | `config.yaml → tts.sample_rate` 尝试 16000 / 24000 / 32000 |
| F8 热键无效 | 用管理员身份运行，或控制台输入 `agent <名字>` |
| 识别不准 | 调整 `audio.vad_mode`（0~3）；确认系统麦克风权限 |
| 被自己的声音误打断 | 戴耳机，或调大 `audio.barge_in_frames` |
| 窗口模式截图不可用 | 整屏模式（默认）不依赖 capturekit；窗口模式失败会自动回退 mss |

## 免责声明

本项目面向个人在自己设备上的个人助手场景。使用屏幕捕获功能时，请遵守当地法律法规及游戏/平台服务条款，勿录制他人隐私内容，勿用于任何破坏游戏公平性的用途。调用 MIMO API 产生的费用由使用者本人的账号承担。

## License

MIT，详见 [LICENSE](LICENSE)。欢迎 Issue 和 PR。
