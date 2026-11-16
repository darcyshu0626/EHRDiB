"""Run the workflow; Slurm jobs also verify and prepare the Conda environment."""

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

ENVIRONMENT_CHECK = """
import sys
from importlib import import_module
from importlib.metadata import distribution, version
from pathlib import Path

for name in (
    'lightgbm', 'lifelines', 'pandas', 'numpy', 'scipy', 'sklearn', 'torch', 'yaml', 'ehrdib.run',
    'datasets.duke.preprocessing', 'datasets.duke.evaluation', 'duckdb',
    'datasets.mimic.preprocessing', 'datasets.mimic.evaluation',
):
    import_module(name)
assert sys.version_info[:2] == (3, 10), 'Python 3.10 is required'
assert import_module('torch').ones(1, device='cpu').sum().item() == 1, 'PyTorch CPU execution failed'
assert int(version('setuptools').split('.')[0]) >= 68, 'setuptools >= 68 is required'
distribution('wheel')
distribution('ehrdib')
root = Path(sys.argv[1]).resolve()
for name in ('ehrdib', 'datasets'):
    module = import_module(name)
    assert Path(module.__file__).resolve().parent.parent == root, (
        'Wrong editable project installation'
    )
"""


def _check_environment(python):
    for arguments in (("-c", ENVIRONMENT_CHECK, str(ROOT)), ("-m", "pip", "check")):
        result = subprocess.run(
            [*python, *arguments], cwd=ROOT, capture_output=True, text=True
        )
        if result.returncode:
            print(result.stdout + result.stderr, file=sys.stderr, flush=True)
            return False
    return True


def _prepare_environment(conda, prefix, python):
    import fcntl

    signature = hashlib.sha256(str(ROOT).encode())
    for name in ("environment.yaml", "pyproject.toml"):
        signature.update((ROOT / name).read_bytes())
    expected = signature.hexdigest()
    stamp = prefix / ".ehrdib-environment"
    prefix.parent.mkdir(parents=True, exist_ok=True)

    # Both datasets and devices share one environment; serialize installation only.
    with prefix.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (
            stamp.is_file()
            and stamp.read_text().strip() == expected
            and (prefix / "bin" / "python").is_file()
            and _check_environment(python)
        ):
            print("  Environment ready", flush=True)
            return
        action = "update" if (prefix / "conda-meta").is_dir() else "create"
        print(f"  Preparing environment: conda env {action}", flush=True)
        command = [
            conda, "env", action, "--prefix", str(prefix),
            "--file", str(ROOT / "environment.yaml"),
        ]
        if action == "create":
            command.append("--yes")
        subprocess.run(
            command, cwd=ROOT, check=True,
            env={**os.environ, "CONDA_ALWAYS_YES": "true", "PIP_NO_INPUT": "1"},
        )
        if not _check_environment(python):
            raise RuntimeError("Environment verification failed after installation; job cancelled")
        stamp.write_text(expected + "\n")


def _run_slurm(arguments):
    conda = shutil.which(os.environ.get("CONDA_EXE") or "conda")
    if not conda:
        raise RuntimeError("Conda is unavailable; activate Conda before running main.py")
    prefix = Path(os.environ["EHRDIB_ENV"])
    libraries = str(prefix / "lib")
    if os.environ.get("LD_LIBRARY_PATH"):
        libraries += os.pathsep + os.environ["LD_LIBRARY_PATH"]
    python = [
        conda, "run", "--no-capture-output", "--prefix", str(prefix),
        "env", f"LD_LIBRARY_PATH={libraries}", "python", "-I", "-B",
    ]
    print("\n" + "=" * 60 + f"\nEnvironment: {prefix.name}\n" + "=" * 60, flush=True)
    _prepare_environment(conda, prefix, python)
    # Dispatch inside the verified environment without bootstrapping a second time.
    # Device validation remains in the workflow, separate from dependency repair.
    return subprocess.run([*python, "-u", "-m", "ehrdib", *arguments]).returncode


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] not in ("-h", "--help") and arguments[0].startswith("-"):
        arguments.insert(0, "run")
    if os.name == "posix":
        os.umask(0o077)
    threads = os.environ.get("SLURM_CPUS_PER_TASK")
    if threads:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = threads
    try:
        if os.environ.get("SLURM_JOB_ID"):
            return _run_slurm(arguments)
        # Local source runs use the caller's environment, as before.
        from ehrdib.run import main as run

        return run(arguments)
    except subprocess.CalledProcessError as exc:
        print(f"ehrdib: environment installation failed (exit {exc.returncode})", file=sys.stderr)
        return exc.returncode if exc.returncode > 0 else 1
    except (OSError, RuntimeError) as exc:
        print(f"ehrdib: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
