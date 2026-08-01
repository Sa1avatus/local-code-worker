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


def test_apply_unified_hunks_creates_new_file_from_zero_range() -> None:
    assert apply_unified_hunks("", "@@ -0,0 +1,2 @@\n+one\n+two\n", "new.py") == "one\ntwo\n"


def test_apply_unified_hunks_normalizes_git_headers_and_crlf() -> None:
    diff = "diff --git a/a.py b/a.py\r\n--- a/a.py\r\n+++ b/a.py\r\n@@ -1 +1 @@\r\n-one\r\n+two\r\n"
    assert apply_unified_hunks("one\n", diff, "a.py") == "two\n"


def test_apply_unified_hunks_rejects_complete_source_as_patch() -> None:
    with pytest.raises(PatchValidationError, match="Missing unified diff hunk header"):
        apply_unified_hunks("", "VALUE = 1\n", "a.py")
