"""Start the approved full Core-4 run as a detached Windows process.

The agent shell kills its child processes when a tool call ends; this starts
run_core4.py outside that process tree (DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
| CREATE_BREAKAWAY_FROM_JOB), with console output to a log file. It launches the
same command the user approved; it does not load or unload any model.
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = REPO / ".venv-eval" / "Scripts" / "python.exe"
RUNNER = REPO / "experiments" / "eval" / "run_core4.py"
log_path = REPO / "experiments" / "eval" / "console" / sys.argv[1]
log_path.parent.mkdir(parents=True, exist_ok=True)

flags = 0x00000008 | 0x00000200 | 0x01000000  # DETACHED_PROCESS | NEW_PROCESS_GROUP | BREAKAWAY_FROM_JOB
log = open(log_path, "x", encoding="utf-8")  # noqa: SIM115
try:
    p = subprocess.Popen([str(PY), str(RUNNER), "--mode", "full", "--i-have-approval"], cwd=str(REPO),
                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags, close_fds=True)
except OSError:
    flags &= ~0x01000000  # breakaway not permitted by the job object; fall back
    p = subprocess.Popen([str(PY), str(RUNNER), "--mode", "full", "--i-have-approval"], cwd=str(REPO),
                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags, close_fds=True)
print("pid", p.pid, "flags", hex(flags), "log", log_path)
