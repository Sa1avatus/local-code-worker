import re

from .exceptions import PatchValidationError

HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def apply_unified_hunks(original: str, diff: str, display_path: str) -> str:
    """Apply a headerless unified diff to UTF-8 text without invoking a shell."""
    source = original.splitlines(keepends=True)
    lines = diff.splitlines(keepends=True)
    result: list[str] = []
    source_index = 0
    index = 0
    hunk_count = 0
    while index < len(lines):
        header = HUNK_HEADER.match(lines[index].rstrip("\r\n"))
        if header is None:
            raise PatchValidationError(f"Invalid unified diff header for {display_path}")
        old_start, old_count = int(header.group(1)), int(header.group(2) or 1)
        if old_start < 1 or old_count < 0:
            raise PatchValidationError(f"Invalid unified diff range for {display_path}")
        target_index = old_start - 1
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
