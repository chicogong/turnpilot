# 独立动作闸门与无原文事件回放 / Standalone action replay

这是一个**可选、离线的因果机制验证**，不是宿主接线或真实设备采集器。`ProvisionalActionGate` 把上游声学候选视为“可以开始判断”，而非“立刻回应”。它复用 `TurnPolicy` 决定动作类型，不运行 ASR/Jev/LLM/TTS，不控制播放，也不修改固定 640 ms 参考策略的默认值。

## 决策顺序

1. 上游提供 `TurnRef`、同一单调时钟的声学观察，并显式 `arm()` 候选；无候选时不由停顿自动放行动作。
2. 同时刻若有续说，续说优先、候选撤销。助手正在播放时，近端/回声条件由原策略快速判断是否 `YIELD_ASSISTANT`，不等候 Jev。
3. 候选后、最大停顿前，只有**当前轮、当前 ASR 修订、实际已到达、未过期**的高完整度语义分数，加上最终 ASR，才能提前放行原策略给出的回应/澄清/忽略动作。部分 ASR、高分但修订过期、仅声学 640 ms 回退都继续等待。
4. 到 `PolicyConfig.max_pause_ms`（默认 1200 ms）时，让原策略按清晰度、是否有字、语义等决定动作；最多建议一次。需要新鲜声学观察才能在这个时点判断；没有实时计时器或新输入时，回放不会凭空执行动作。宿主仍须检查 `decision_is_current()` 和播放所有权。

这会用等待换取潜在的错误行动下降，**不是免费提速**。若最终 ASR/语义没有在截止前到达，默认对照可在 640 ms 建议动作，而这个可选闸门要等到 1200 ms；两者都有及时新观察时，回退点相差 560 ms。这不是已测得的设备时延；若声学/时钟事件缺失，还可能更晚或根本不行动。没有人工动作标签，不能说它改善了误回应。

## 无原文事件格式

`TraceRecorder` 只在调用方显式 `append()` 时在内存收集；`to_jsonl()` 由调用方自行决定是否持久化。它不是录麦克风的程序，也不会自动接入 ASR、播放或 Jev。每条 JSONL 行必须恰有 `schema=1`、`kind`、`at_ms`、`session_id`、`turn_id`、`generation`、`data`。ID 只能是最多 64 个 ASCII 字母/数字/`_`/`-` 的不含姓名的假名；即使没有原文，时间线和假名仍可能是敏感元数据，真实记录须获授权并放在 Git 外。

| `kind` | `data` 的精确字段 | 含义 |
| --- | --- | --- |
| `turn_start`、`stop`、`candidate`、`tick` | `{}` | 新轮次、停止、上游声学候选、显式推进时钟 |
| `acoustic` | `speech_active`、`near_end_speech`、`echo_likely`、`speech_duration_ms`、`pause_duration_ms`、`audio_quality` | 上游 VAD/回声提示与停顿状态；不包含波形 |
| `asr` | `revision`、`has_text`、`is_final` | 修订**实际可用时刻**；不包含转写。`has_text` 仅代表非空，不说明内容正确 |
| `semantic` | `transcript_revision`、`complete_probability`、`clarification_probability`、`backchannel_probability`、`response_probability` | 调用方已算好的分数到达时刻；最后一项可为 `null`；不调用远端 |
| `playback` | `speaking` | 助手播放状态边沿；不是播放确认或取消账本 |

额外字段如 `text`、`transcript`、`audio_path`，错误类型、重复 JSON 键、NaN/Inf、倒序时钟、超大文件会被拒绝。上限是 10,000 事件、2 MB 文件和 4 KB 单行；CLI 错误只给通用信息，不回显内容。回放用固定的合成占位字代表 `has_text=true`，因此它**无法**测试 ASR 内容、Jev 正确率、音频是否清楚或用户真正意图。

```bash
uv run --no-sync turnpilot-action-replay examples/synthetic-action-trace.jsonl
uv run --no-sync turnpilot-action-replay examples/synthetic-action-trace.jsonl --compare-direct-asr
```

[合成范例](../examples/synthetic-action-trace.jsonl)包含一次最终 ASR/高分后的建议回应，以及一次候选后续说撤销。CLI 只输出动作数量、候选/撤销/过期事件计数与 `synthetic_or_declared_trace_only` 证据级别，不打印 ID 或文本。回放按事件到达时间处理；同一时刻停止与续说优先，旧轮次和不匹配的修订不能授权动作。

`--compare-direct-asr` 在**同一条时间线**上额外回放一个刻意简化的基线：每个活跃轮次的第一条非空 final ASR 一到，就建议一次回应；同刻 `stop` 优先，旧轮次/旧修订被拒绝，但不检查声学续说、语义或播放状态。合成范例里，这个基线在 1350 和 2600 ms 建议两次 `COMMIT`（第二次仍在续说）；可选闸门只在 1360 ms 建议一次。首轮闸门因此晚 10 ms。这只说明防护规则在**构造事件**中起作用，不代表所有直接接 ASR 的产品，也不代表真实错误率或设备时延。[完整对照和公开语料限制](evaluation.md)另有说明。

## 验证边界与下一关

单元测试覆盖：部分/最终 ASR、修订不匹配、老轮次、同时间续说、640 ms 回退延后、1200 ms 上限、音频与语义澄清、播放中打断、停止、格式与隐私约束。CI 在四个 Python 版本运行合成 CLI 烟测。它证明的是**实现遵守这些规则**，不是“真实对话更自然”。

要判断是否值得用，仍须先取得获授权的连续设备对话及同钟声学、真实 ASR 修订可用时刻、播放边沿、人工续说/动作标签；按说话人和设备留出，以相同事件时间线对照未改动的宿主策略与可选闸门，同时报告错误行动、误打断、提前截断及完成后 P50/P95/P99。缺一层就保留未验收，且不接入宿主或切换默认。数据结构与人工审计门槛见[真实对话证据入口](real-conversation-evidence.md)。
