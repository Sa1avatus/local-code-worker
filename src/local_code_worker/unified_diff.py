import re

from .exceptions import PatchValidationError

HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def normalize_unified_diff(diff: str, display_path: str) -> str:
    """Normalize safe presentation wrappers without interpreting source text as a diff."""
    lines = diff.replace("\r\n", "\n").replace("\r", "\n").splitlines(keepends=True)
    if lines and lines[0].strip() in {"```diff", "```patch", "```"}:
        if len(lines) < 2 or lines[-1].strip() != "```":
            raise PatchValidationError(f"Unclosed Markdown fence for {display_path}")
        lines = lines[1:-1]
    while lines and (
        lines[0].startswith("diff --git ")
        or lines[0].startswith("--- ")
        or lines[0].startswith("+++ ")
    ):
        lines.pop(0)
    if not lines or not lines[0].startswith("@@ "):
        first_line = lines[0].rstrip("\n") if lines else "<empty>"
        raise PatchValidationError(
            f"Missing unified diff hunk header for {display_path}; got {first_line!r}; "
            "expected '@@ -old,count +new,count @@'"
        )
    return "".join(lines)


def apply_unified_hunks(original: str, diff: str, display_path: str) -> str:
    """Apply a headerless unified diff to UTF-8 text without invoking a shell."""
    source = original.splitlines(keepends=True)
    lines = normalize_unified_diff(diff, display_path).splitlines(keepends=True)
    result: list[str] = []
    source_index = 0
    index = 0
    hunk_count = 0
    while index < len(lines):
        header = HUNK_HEADER.match(lines[index].rstrip("\r\n"))
        if header is None:
            raise PatchValidationError(f"Invalid unified diff header for {display_path}")
        old_start, old_count = int(header.group(1)), int(header.group(2) or 1)
        is_new_file_hunk = old_start == 0 and old_count == 0 and not source and source_index == 0
        if (old_start < 1 and not is_new_file_hunk) or old_count < 0:
            raise PatchValidationError(f"Invalid unified diff range for {display_path}")
        target_index = 0 if is_new_file_hunk else old_start - 1
        if target_index < source_index or target_index > len(source):
            raise PatchValidationError(f"Unified diff range does not match {display_path}")
        result.extend(source[source_index:target_index])
        source_index = target_index
        index += 1
        consumed = 0
        while index < len(lines) and not lines[index].startswith("@@ "):
            line = lines[index]
            if not line or line[0] not in {" ", "+", "-", "\\"}:
                raise PatchValidationError(f"Invalid unified diff line for {display_path}")
            marker, text = line[0], line[1:]
            if marker == "\\":
                index += 1
                continue
            if marker in {" ", "-"}:
                if source_index >= len(source) or source[source_index] != text:
                    raise PatchValidationError(
                        f"Unified diff context does not match {display_path}"
                    )
                source_index += 1
                consumed += 1
            if marker in {" ", "+"}:
                result.append(text)
            index += 1
        if consumed != old_count:
            raise PatchValidationError(f"Unified diff line count does not match {display_path}")
        hunk_count += 1
    if not hunk_count:
        raise PatchValidationError(f"Unified diff has no hunks for {display_path}")
    result.extend(source[source_index:])
    return "".join(result)
