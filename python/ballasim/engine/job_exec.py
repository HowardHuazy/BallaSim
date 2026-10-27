"""The supervisor of a job's ``exec`` step (:mod:`ballasim.engine.job`): ``python -I -S job_exec.py PARENT -- ARGV``.

The runner starts this script in a new process group and sends a cancel's SIGINT to that group once, so the program
and its own children each get it once, never also the client's group signal. This process ignores that SIGINT and
waits for the program. When the runner dies, also by SIGKILL, it kills the whole group (Linux: PR_SET_PDEATHSIG), so
no program or grandchild outlives the job. Exit status: the program's, or 128 + N when signal N ended it.

Standard library only, run by path: the runner starts it isolated (``-I -S``), so it imports nothing from the job
directory or from PYTHONPATH. The program itself gets the runner's environment unchanged."""

import os
import signal
import subprocess
import sys


def _kill_group(_signum=None, _frame=None) -> None:
    os.killpg(0, signal.SIGKILL)


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] != "--":
        print("usage: python -I -S job_exec.py PARENT_PID -- PROGRAM [ARGS]", file=sys.stderr)
        return 2
    parent, program = int(argv[0]), argv[2:]
    # A handler, not SIG_IGN: exec resets handlers, so the program starts with the default SIGINT action.
    signal.signal(signal.SIGINT, lambda _signum, _frame: None)
    if sys.platform.startswith("linux"):
        import ctypes

        signal.signal(signal.SIGHUP, _kill_group)
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGHUP)  # PR_SET_PDEATHSIG
    if os.getppid() != parent:  # the runner died before the prctl
        _kill_group()
    try:
        proc = subprocess.Popen(program)
    except OSError as e:
        print(f"exec: {program[0]}: {e}", file=sys.stderr)
        return 127
    rc = proc.wait()
    return 128 - rc if rc < 0 else rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
