import pytest

from local_code_worker.exceptions import PatchValidationError
from local_code_worker.unified_diff import apply_unified_hunks


def test_apply_unified_hunks_changes_only_requested_lines() -> None:
    original = "one\ntwo\nthree\n"
    diff = "@@ -1,3 +1,3 @@\n one\n-two\n+updated\n three\n"

    assert apply_unified_hunks(original, diff, "src/example.py") == "one\nupdated\nthree\n"


def test_apply_unified_hunks_rejects_mismatched_context() -> None:
    with pytest.raises(PatchValidationError, match="context does not match"):
        apply_unified_hunks("one\n", "@@ -1 +1 @@\n-wrong\n+two\n", "src/example.py")
