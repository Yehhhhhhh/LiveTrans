"""Static checks run by CI (.github/workflows/ci.yml) and runnable locally.

    python .github/scripts/lint.py

The project has no linter configured, so this only checks what the repository
already promises:
  1. every Python file under backend/ compiles (syntax check, no imports run);
  2. every first-party frontend JavaScript file parses (`node --check`);
  3. every .bat file is pure ASCII with CRLF line endings, as the comment at
     the top of install.bat / start.bat requires (cmd.exe misparses UTF-8).

Exits non-zero and prints one `file: problem` line per failure.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IN_CI = os.environ.get("GITHUB_ACTIONS") == "true"


def error(path: Path, msg: str) -> None:
    rel = path.relative_to(ROOT).as_posix()
    # GitHub turns ::error lines into annotations on the PR's Files tab.
    if IN_CI:
        print(f"::error file={rel}::{msg.splitlines()[0] if msg else 'error'}")
    print(f"{rel}: {msg}")


def check_python() -> int:
    failures = 0
    for path in sorted((ROOT / "backend").rglob("*.py")):
        if any(part in {".venv", "__pycache__"} for part in path.parts):
            continue
        try:
            compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        except (SyntaxError, ValueError) as e:
            where = f" (line {e.lineno})" if getattr(e, "lineno", None) else ""
            error(path, f"Python syntax error{where}: {e}")
            failures += 1
    return failures


def check_javascript() -> int:
    files = sorted(p for p in (ROOT / "frontend").glob("*.js"))  # vendor/ is third-party
    if not files:
        return 0
    node = shutil.which("node")
    if node is None:
        print("node not found on PATH; install Node.js 22+ to check frontend JavaScript", file=sys.stderr)
        return 1
    failures = 0
    for path in files:
        r = subprocess.run([node, "--check", str(path)], capture_output=True, text=True)
        if r.returncode != 0:
            error(path, f"JavaScript syntax error:\n{r.stderr.strip()}")
            failures += 1
    return failures


def check_bat() -> int:
    failures = 0
    for path in sorted(ROOT.glob("*.bat")):
        data = path.read_bytes()
        bad = [i for i, b in enumerate(data) if b > 0x7F]
        if bad:
            line = data[: bad[0]].count(b"\n") + 1
            error(path, f"non-ASCII byte on line {line}; .bat files must be pure ASCII")
            failures += 1
        lone_lf = data.replace(b"\r\n", b"").count(b"\n")
        if lone_lf:
            error(path, f"{lone_lf} line(s) end in LF instead of CRLF")
            failures += 1
    return failures


def main() -> int:
    checks = [("Python syntax", check_python), ("JavaScript syntax", check_javascript), (".bat encoding", check_bat)]
    total = 0
    for name, fn in checks:
        n = fn()
        print(f"[{'FAIL' if n else ' OK '}] {name}")
        total += n
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
