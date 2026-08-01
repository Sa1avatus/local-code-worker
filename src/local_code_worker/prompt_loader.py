from pathlib import Path

from .exceptions import WorkerError
from .models import ProposalFormat


def load_system_prompt(worker_root: Path, proposal_format: ProposalFormat) -> str:
    prompt_name = (
        "implementation_patch_system.txt"
        if proposal_format is ProposalFormat.PATCH
        else "implementation_system.txt"
    )
    prompt_path = worker_root / "prompts" / prompt_name
    try:
        return prompt_path.read_text(encoding="utf-8")
    except OSError as error:
        raise WorkerError(f"Cannot read system prompt: {prompt_path}") from error
