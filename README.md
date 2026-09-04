# CPCodeAgent

CPCodeAgent 是一个使用 Python 实现的轻量级编程智能体框架。它可以在指定工作区中读取和修改代码、执行命令、运行验证，并通过持久化日志安全地完成多轮任务。

项目强调简洁、可恢复和可审计，适合研究编程智能体的核心机制，也可以作为小型代码 Agent 的实现基础。

## 核心功能

- **多轮会话**：会话状态持久保存，可使用会话 ID 随时恢复。
- **代码操作**：内置文件读取、搜索、编辑、写入和命令执行工具。
- **安全恢复**：工具调用采用 `INTENT → STARTED → COMMITTED` 流程，中断后根据实际文件状态决定提交、重试或停止。
- **权限控制**：统一管理读取、工作区写入、网络和外部副作用，可设置允许、询问或拒绝。
- **并发调度**：连续的只读工具可并发执行，写操作通过串行屏障保证顺序。
- **上下文压缩**：完整事件保存在 Journal 中，模型上下文可按窗口压力分层压缩和重建。
- **任务规划**：复杂任务可以维护可见、可恢复的执行计划，避免未完成就提前结束。
- **子智能体**：支持只读调研和隔离修改，补丁由主智能体检查后显式应用。
- **记忆与 Skills**：支持用户级、会话级 Markdown 记忆和按需加载的 `SKILL.md`。
- **流式终端**：实时显示模型输出、工具进度和验证结果，并支持回退模型。

## 设计亮点

### 可恢复的执行日志

Journal 是追加写入的持久化事实源，记录模型响应、工具意图、执行结果和工作区版本。模型上下文只是从 Journal 构建出的临时视图，因此会话可以在启动、中断或上下文溢出后重建。

文件写入恢复时会比较执行前后的内容摘要：已达到目标状态则补记提交，仍是原状态则安全重试，出现其他状态则报告冲突。无法验证结果的命令不会被盲目重放。详细设计见 [崩溃一致性说明](docs/crash-consistency.md)。

### 隔离的子智能体

`inspect` 子任务共享工作区，但只能读取；`patch` 子任务在工作区副本中修改。子智能体拥有独立上下文和 Journal，不能继续创建下一层子智能体，修改结果也不会自动合并到主工作区。

### 清晰的扩展接口

项目通过少量接口扩展能力：`Model`、`Tool`、`Policy`、`Executor`、`SkillRegistry` 和 `Verifier`，没有引入复杂的通用 Agent Graph。

## 快速开始

### 1. 安装

需要 Python 3.11 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

### 2. 配置模型

```bash
cp .env.example .env
```

编辑 `.env`，至少填写：

```dotenv
OPENAI_API_KEY=your-api-key
CPCODEAGENT_MODEL=gpt-4.1
```

使用 OpenAI API 兼容服务时，额外设置：

```dotenv
OPENAI_BASE_URL=https://your-provider.example/v1
```

配置优先级为：命令行参数 > Shell 环境变量 > `.env` > 内置默认值。

### 3. 启动

处理当前目录：

```bash
cpcodeagent
```

指定其他项目：

```bash
cpcodeagent --workspace /path/to/your/project
```

执行一次性任务：

```bash
cpcodeagent "检查项目并修复失败的测试" --workspace .
```

在最终回答前运行验证：

```bash
cpcodeagent "实现需求并确保测试通过" \
  --workspace . \
  --verify "python -m pytest"
```

### 4. 恢复会话

```bash
cpcodeagent --resume <session-id>
```

Journal 默认保存在 `~/.cpcodeagent/runs/<session-id>.jsonl`。恢复时会沿用会话原有的工作区、权限策略和执行器。

## 执行器与权限

默认的 `LocalExecutor` 适合可信项目。内置文件工具只能访问工作区，但本地命令不属于操作系统级沙箱。

对于不熟悉的项目，可以使用 Docker：

```bash
cpcodeagent \
  --workspace /path/to/your/project \
  --executor docker \
  --docker-image python:3.12-slim
```

Docker 执行器默认关闭网络，并限制 capabilities、CPU、内存和进程数。所选镜像需要包含目标项目运行所需的环境。

常用权限参数：

| 参数 | 作用 |
| --- | --- |
| `--read-only` | 禁止工作区写入 |
| `--allow-host HOST` | 允许访问指定网络目标，可重复使用 |
| `--external-writes allow\|ask\|deny` | 设置外部写操作策略 |

## 交互命令

| 命令 | 说明 |
| --- | --- |
| `/status` | 查看会话状态 |
| `/memory [user\|session]` | 查看持久记忆 |
| `/remember <user\|session> <内容>` | 保存记忆 |
| `/forget <user\|session> <key\|all>` | 删除记忆 |
| `/help` | 查看帮助 |
| `/exit` | 保存并退出 |

## Skills

项目级 Skill 放在 `.cpcodeagent/skills/<name>/SKILL.md`，个人 Skill 放在 `~/.cpcodeagent/skills/<name>/SKILL.md`。启动时只加载名称和描述，使用时才读取完整说明。Skill 可以声明所需工具，但不能扩大会话权限。

示例见 [examples/skills/debugging/SKILL.md](examples/skills/debugging/SKILL.md)。

## 架构概览

```text
Session
   ├── Turns         INPUT → THINK → ACT → CHECK → FINAL
   ├── Memory        用户级 + 会话级 Markdown 记忆
   └── Journal       完整历史与恢复状态
          ├── Context       计划、记忆与压缩历史
          ├── Runtime       动作分类、权限、执行与恢复
          └── Subagent      独立上下文与隔离工作区
```

核心代码位于 `cpcodeagent/`：

- `kernel.py`：主运行循环和预算；
- `context.py`：上下文构建与压缩；
- `tools.py` / `builtin_tools.py`：工具协议和内置工具；
- `journal.py` / `recovery.py`：持久日志和恢复；
- `subagents.py`：子智能体与补丁；
- `executor.py`：本地和 Docker 执行器；
- `memory.py` / `skills.py`：记忆与 Skills。

## 开发与测试

```bash
python -m examples.offline_demo
python -m pytest
ruff check cpcodeagent tests
```

查看全部参数：

```bash
cpcodeagent --help
```
