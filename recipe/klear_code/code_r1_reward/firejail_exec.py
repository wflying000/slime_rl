"""Sandboxed Python executor used by the klear code reward.

Primary path: ``firejail`` (reproduces KlearReasoner behavior exactly).
Fallback path: plain ``subprocess.run`` with ``resource.setrlimit`` guards —
triggered automatically when ``firejail`` is not on ``$PATH`` so the reward
works on macOS dev machines and containers without firejail installed.

Both paths share the same ``(success: bool, output_or_error: str)`` return
shape and the same error-message prefix used by klear's scorer. Execution
timeouts raise RewardTimeout so the sample is excluded from training.
"""

from __future__ import annotations

from recipe.rl_reward.timeout_utils import RewardTimeout

import os
import platform
import shutil
import signal
import subprocess
import threading
import logging
from tempfile import NamedTemporaryFile, TemporaryDirectory

_ERROR_MSG_PREFIX = "Failed to execute program: "
_DEFAULT_TIMEOUT_SECONDS = 35

# Fallback memory cap (MiB). Matches --rlimit-as=1024m in the firejail profile.
_FALLBACK_MEM_LIMIT_MB = 1024
# Fallback CPU-time cap (s). Generous multiplier on wall-clock timeout.
_FALLBACK_CPU_LIMIT_FACTOR = 3
# Cap code-execution subprocesses per reward worker. The rollout batch can
# schedule many klear_code rewards at once, so local per-sample thread limits
# are not enough to protect host memory.
_MAX_CODE_EXEC_CONCURRENCY = int(os.environ.get("KLEAR_CODE_EXEC_MAX_CONCURRENCY", "4"))
_ALLOW_SUBPROCESS_FALLBACK = os.environ.get("KLEAR_CODE_ALLOW_SUBPROCESS_FALLBACK", "0") == "1"

_FIREJAIL_BIN = shutil.which("firejail")
_IS_LINUX = platform.system() == "Linux"
_EXEC_SEMAPHORE = threading.BoundedSemaphore(max(1, _MAX_CODE_EXEC_CONCURRENCY))
_BACKEND_LOGGED = False
_BACKEND_LOG_LOCK = threading.Lock()
logger = logging.getLogger(__name__)


def _limit_wrapper_source(timeout: int) -> str:
    mem_bytes = _FALLBACK_MEM_LIMIT_MB * 1024 * 1024
    cpu_seconds = max(1, timeout * _FALLBACK_CPU_LIMIT_FACTOR)
    return f"""\
import runpy
import sys

try:
    import resource

    def _set_limit(name, value):
        limit = getattr(resource, name, None)
        if limit is None:
            return
        try:
            resource.setrlimit(limit, (value, value))
        except (ValueError, OSError):
            pass

    _set_limit("RLIMIT_AS", {mem_bytes})
    _set_limit("RLIMIT_CPU", {cpu_seconds})
    _set_limit("RLIMIT_FSIZE", {8 * 1024 * 1024})
    _set_limit("RLIMIT_NOFILE", {32})
    _set_limit("RLIMIT_NPROC", {32})
except Exception:
    pass

if len(sys.argv) < 2:
    raise SystemExit("missing target")

target = sys.argv[1]
args = sys.argv[2:]
if target == "-m":
    if not args:
        raise SystemExit("missing module")
    module = args[0]
    sys.argv = [module] + args[1:]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
else:
    sys.argv = [target] + args
    runpy.run_path(target, run_name="__main__")
"""


def _write_limit_wrapper(path: str, timeout: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(_limit_wrapper_source(timeout))


def _make_preexec_fn(memory_mb: int, cpu_seconds: int):
    """Return a preexec_fn that sets RLIMIT_AS and RLIMIT_CPU where supported."""

    def _preexec():
        try:
            import resource

            mem_bytes = memory_mb * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            except (ValueError, OSError):
                pass
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            except (ValueError, OSError):
                pass
            for limit_name, value in (
                ("RLIMIT_FSIZE", 8 * 1024 * 1024),
                ("RLIMIT_NOFILE", 32),
                ("RLIMIT_NPROC", 32),
            ):
                limit = getattr(resource, limit_name, None)
                if limit is None:
                    continue
                try:
                    resource.setrlimit(limit, (value, value))
                except (ValueError, OSError):
                    pass
        except Exception:
            pass
        # Detach into a new process group so we can kill the whole tree.
        try:
            os.setsid()
        except OSError:
            pass

    return _preexec


def _build_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for k, v in os.environ.copy().items():
        if k[:4] != "KML_":
            env[k] = v
    env["OPENBLAS_NUM_THREADS"] = "1"
    env.pop("PYTHONPATH", None)
    return env


def _kill_process_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass


def _log_backend_once(backend: str) -> None:
    global _BACKEND_LOGGED
    if _BACKEND_LOGGED:
        return
    with _BACKEND_LOG_LOCK:
        if _BACKEND_LOGGED:
            return
        logger.warning("klear_code sandbox backend: %s", backend)
        _BACKEND_LOGGED = True


def _run_command(
    command: list[str],
    *,
    input_bytes: bytes | None = None,
    cwd: str | None = None,
    env: dict[str, str],
    timeout: int,
    preexec_fn=None,
) -> subprocess.CompletedProcess[bytes] | None:
    """Run a command in a new process group and always reap/kill it."""
    proc: subprocess.Popen[bytes] | None = None
    try:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.PIPE if input_bytes is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            preexec_fn=preexec_fn,
            start_new_session=(preexec_fn is None),
        )
        stdout, stderr = proc.communicate(input=input_bytes, timeout=timeout)
        return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        if proc is not None:
            _kill_process_group(proc)
            try:
                stdout, stderr = proc.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                stdout, stderr = b"", b""
        raise RewardTimeout(f"code execution exceeded {timeout}s")
    finally:
        if proc is not None:
            _kill_process_group(proc)
            try:
                proc.wait(timeout=3)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                pass


def _run_with_firejail(
    code: str,
    stdin: str | None,
    timeout: int,
    pytest: str | None,
    env: dict[str, str],
) -> tuple[bool, str]:
    """Original klear firejail path."""
    command = [
        _FIREJAIL_BIN or "firejail",
        "--private",
        "--quiet",
        "--seccomp=socket",
        "--profile=pip",
        "--rlimit-nproc=32",
        "--rlimit-nofile=32",
        "--rlimit-fsize=8m",
        "--rlimit-as=1024m",
        # _run_command owns the wall-clock limit and reports RewardTimeout;
        # firejail's own timer otherwise looks like an ordinary failed test.
        "--kill-at-exit",
    ]

    if pytest:
        with TemporaryDirectory() as tmpdir:
            assert stdin is None, "STDIN is not supported with pytest"
            wrapper_path = os.path.join(tmpdir, "_klear_limit_wrapper.py")
            _write_limit_wrapper(wrapper_path, timeout)
            with open(os.path.join(tmpdir, "solution.py"), "w") as f:
                f.write(code)
            with open(os.path.join(tmpdir, "test_solution.py"), "w") as f:
                f.write(pytest)
            command.insert(4, f"--whitelist={tmpdir}")
            command.extend(["python", wrapper_path, "-m", "pytest", tmpdir])
            result = _run_command(command, cwd=tmpdir, env=env, timeout=timeout + 3)
            if result is None:
                return False, _ERROR_MSG_PREFIX + "Execution timeout, the process is forcibly terminated"
    else:
        with NamedTemporaryFile() as tmp:
            with NamedTemporaryFile(mode="w", suffix=".py") as wrapper:
                tmp.write(code.encode())
                tmp.flush()
                _write_limit_wrapper(wrapper.name, timeout)
                command.insert(4, f"--whitelist={tmp.name}")
                command.insert(4, f"--whitelist={wrapper.name}")
                command.extend(["python", wrapper.name, tmp.name])
                result = _run_command(
                    command,
                    input_bytes=stdin.encode() if stdin else None,
                    env=env,
                    timeout=timeout,
                )
                if result is None:
                    return False, _ERROR_MSG_PREFIX + "Execution timeout, the process is forcibly terminated"

    stderr = result.stderr.decode(errors="replace").strip()
    stdout = result.stdout.decode(errors="replace")
    if result.returncode == 0:
        return True, stdout
    return False, _ERROR_MSG_PREFIX + f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"


def _run_with_subprocess(
    code: str,
    stdin: str | None,
    timeout: int,
    pytest: str | None,
    env: dict[str, str],
) -> tuple[bool, str]:
    """Fallback path used when firejail is unavailable.

    Safety is weaker than firejail — no filesystem isolation, no seccomp —
    but ``RLIMIT_AS``/``RLIMIT_CPU`` and the wall-clock timeout still bound
    each process. Intended for dev/test environments.
    """
    preexec = _make_preexec_fn(
        memory_mb=_FALLBACK_MEM_LIMIT_MB,
        cpu_seconds=timeout * _FALLBACK_CPU_LIMIT_FACTOR,
    ) if _IS_LINUX else None

    if pytest:
        with TemporaryDirectory() as tmpdir:
            assert stdin is None, "STDIN is not supported with pytest"
            with open(os.path.join(tmpdir, "solution.py"), "w") as f:
                f.write(code)
            with open(os.path.join(tmpdir, "test_solution.py"), "w") as f:
                f.write(pytest)
            result = _run_command(
                ["python", "-m", "pytest", tmpdir],
                cwd=tmpdir,
                env=env,
                timeout=timeout + 3,
                preexec_fn=preexec,
            )
            if result is None:
                return False, _ERROR_MSG_PREFIX + "Execution timeout, the process is forcibly terminated"
    else:
        with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        try:
            result = _run_command(
                ["python", tmp_path],
                input_bytes=stdin.encode() if stdin else None,
                env=env,
                timeout=timeout,
                preexec_fn=preexec,
            )
            if result is None:
                return False, _ERROR_MSG_PREFIX + "Execution timeout, the process is forcibly terminated"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    stderr = result.stderr.decode(errors="replace").strip()
    stdout = result.stdout.decode(errors="replace")
    if result.returncode == 0:
        return True, stdout
    return False, _ERROR_MSG_PREFIX + f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"


def code_exec_firejail(
    code: str,
    stdin: str | None = None,
    timeout: int = _DEFAULT_TIMEOUT_SECONDS,
    pytest: str | None = None,
) -> tuple[bool, str]:
    """Execute ``code`` in the sandbox.

    Returns ``(success, output)``. When ``success`` is ``False`` the output
    starts with ``_ERROR_MSG_PREFIX`` (klear compatibility).
    """
    env = _build_env()
    with _EXEC_SEMAPHORE:
        if _FIREJAIL_BIN and _IS_LINUX:
            _log_backend_once(f"firejail ({_FIREJAIL_BIN})")
            return _run_with_firejail(code, stdin, timeout, pytest, env)
        if _IS_LINUX and not _ALLOW_SUBPROCESS_FALLBACK:
            _log_backend_once("missing firejail; fail closed")
            return False, _ERROR_MSG_PREFIX + "firejail is unavailable; refusing unsafe subprocess fallback"
        _log_backend_once("subprocess fallback")
        return _run_with_subprocess(code, stdin, timeout, pytest, env)
