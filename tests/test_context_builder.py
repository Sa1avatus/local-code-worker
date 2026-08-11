from pathlib import Path

from local_code_worker.context_builder import build_context
from local_code_worker.models import PromptFormat, WorkerTask


def task(root: Path, prompt_format: PromptFormat = PromptFormat.XML) -> WorkerTask:
    return WorkerTask(
        task_id="contract-test",
        title="Update code",
        goal="Implement one behavior",
        repository_root=root,
        allowed_files=[Path("src/editable.py")],
        readonly_files=[Path("src/types.py")],
        requirements=["Use the declared type"],
        acceptance_criteria=["Tests pass"],
        validation_commands=[["python", "-m", "pytest", "tests/test_unit.py"]],
        prompt_format=prompt_format,
    )


def test_xml_execution_contract_separates_dependencies_and_task(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "types.py").write_text("class User: pass\n", encoding="utf-8")

    context, statistics = build_context(task(tmp_path))

    assert "<context_dependencies>" in context
    assert '<file path="src/types.py"><![CDATA[class User: pass' in context
    assert "<task_instruction>" in context
    assert "<negative_constraints>" in context
    assert "<output_format>" in context
    assert "<acceptance_criteria>" in context
    assert "<criterion>Tests pass</criterion>" in context
    assert "<validation_commands>" in context
    assert "<argument>pytest</argument>" in context
    assert statistics.paths == ("src/editable.py", "src/types.py")


def test_json_prompt_format_preserves_legacy_context(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "types.py").write_text("TYPE = 1\n", encoding="utf-8")

    context, _ = build_context(task(tmp_path, PromptFormat.JSON))

    assert context.startswith("TASK SPECIFICATION\n\n{")
    assert "BEGIN FILE: src/types.py" in context
