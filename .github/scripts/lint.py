"""Static checks run by CI (.github/workflows/ci.yml) and runnable locally.

    python .github/scripts/lint.py

The project has no linter configured, so this only checks what the repository
already promises:
  1. every Python file under backend/ compiles (syntax check, no imports run);
  2. every first-party frontend JavaScript file parses (`node --check`);
  3. the frontend's unit tests pass (`node --test frontend/tests/*.test.js`);
  4. every .bat file is pure ASCII with CRLF line endings, as the comment at
     the top of install.bat / start.bat requires (cmd.exe misparses UTF-8).

Exits non-zero and prints one `file: problem` line per failure.
"""

from __future__ import annotations

import os
import re
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


def find_node() -> str | None:
    node = shutil.which("node")
    if node is None:
        print("node not found on PATH; install Node.js 22+ to check frontend JavaScript", file=sys.stderr)
    return node


def check_javascript() -> int:
    front = ROOT / "frontend"
    # Subfolders too (lib/, tests/), except vendor/, which is third-party.
    files = sorted(p for p in front.rglob("*.js") if "vendor" not in p.relative_to(front).parts)
    if not files:
        return 0
    node = find_node()
    if node is None:
        return 1
    failures = 0
    for path in files:
        r = subprocess.run([node, "--check", str(path)], capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            error(path, f"JavaScript syntax error:\n{r.stderr.strip()}")
            failures += 1
    return failures


def check_javascript_tests() -> int:
    tests = sorted((ROOT / "frontend" / "tests").glob("*.test.js"))
    if not tests:
        return 0
    node = find_node()
    if node is None:
        return 1
    # TAP output, so the summary ("# pass 12") can be read back.
    r = subprocess.run(
        [node, "--test", "--test-reporter=tap", *map(str, tests)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=ROOT,
    )
    summary = dict(re.findall(r"^# (tests|pass|fail|todo) (\d+)$", r.stdout, re.M))
    if r.returncode != 0:
        error(ROOT / "frontend" / "tests", f"JavaScript unit tests failed:\n{(r.stdout + r.stderr).strip()}")
        return 1
    todo = f", {summary['todo']} todo" if summary.get("todo", "0") != "0" else ""
    print(f"frontend/tests: {summary.get('pass', '?')} of {summary.get('tests', '?')} tests passed{todo}")
    return 0


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
    # Node's output may hold characters the console's code page lacks (e.g. GBK on a Chinese Windows).
    sys.stdout.reconfigure(errors="replace")
    checks = [
        ("Python syntax", check_python),
        ("JavaScript syntax", check_javascript),
        ("JavaScript unit tests", check_javascript_tests),
        (".bat encoding", check_bat),
    ]
    total = 0
    for name, fn in checks:
        n = fn()
        print(f"[{'FAIL' if n else ' OK '}] {name}")
        total += n
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
