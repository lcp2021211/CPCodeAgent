"""Run a complete harness loop without an API key."""

from pathlib import Path
from tempfile import TemporaryDirectory

from cpcodeagent.builtin_tools import build_default_runtime
from cpcodeagent.executor import LocalExecutor
from cpcodeagent.journal import Journal
from cpcodeagent.kernel import Harness
from cpcodeagent.model import ScriptedModel
from cpcodeagent.skills import SkillRegistry
from cpcodeagent.types import ModelResponse, ToolCall


with TemporaryDirectory() as directory:
    workspace = Path(directory)
    skill_root = Path(__file__).parent / "skills"
    skills = SkillRegistry([skill_root])
    runtime = build_default_runtime(skills)
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=(ToolCall("skill-1", "use_skill", {"name": "debugging"}),)),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "write-1",
                        "write_file",
                        {"path": "answer.py", "content": "def answer():\n    return 42\n"},
                    ),
                )
            ),
            ModelResponse(content="Implemented answer.py and observed the write result."),
        ]
    )
    outcome = Harness(model, runtime, skills=skills).run(
        "Create answer.py with an answer() function.",
        LocalExecutor(workspace),
        Journal(workspace / "run.jsonl"),
        "offline-demo",
    )
    print(outcome)
    print((workspace / "answer.py").read_text())

