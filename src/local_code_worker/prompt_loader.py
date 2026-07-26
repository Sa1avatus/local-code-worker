from pathlib import Path

from .exceptions import WorkerError


def load_system_prompt(worker_root: Path) -> str:
    prompt_path = worker_root / "prompts" / "implementation_system.txt"
    try:
        return prompt_path.read_text(encoding="utf-8")
    except OSError as error:
        raise WorkerError(f"Cannot read system prompt: {prompt_path}") from error
