"""Discover the CUDA toolkit that the current environment actually provides.

Nothing in this repo hard-codes a toolkit path. Everything below is derived
from the `nvidia-*` wheels installed in whichever virtualenv is active (or
from a system toolkit, if that is what you have), so the same checkout builds
on any machine and any CUDA version without editing a line.

Usable as a library (see bench.py) or from the shell, for the Makefile:

    python cuda_env.py home    # toolkit root, the thing you'd call CUDA_HOME
    python cuda_env.py lib     # link directory (pip uses lib/, system lib64/)
    python cuda_env.py nvcc    # full path to nvcc
    python cuda_env.py arch    # sm_XX of the GPU in this box
"""

from __future__ import annotations

import functools
import os
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent

# Where generated link-time shims live. Regenerated on demand, safe to delete.
BUILD_DIR = HERE / "build"

DEFAULT_ARCH = "sm_89"  # only used if no GPU can be interrogated


@functools.cache
def cuda_home() -> pathlib.Path:
    """Root of a CUDA toolkit containing bin/nvcc.

    Search order: an explicit $CUDA_HOME/$CUDA_PATH, then the pip toolkit in
    the active environment (`nvidia/cu13`, `nvidia/cu12`, ...), then whatever
    nvcc happens to be on $PATH.
    """
    for var in ("CUDA_HOME", "CUDA_PATH"):
        p = os.environ.get(var)
        if p and (pathlib.Path(p) / "bin" / "nvcc").exists():
            return pathlib.Path(p)

    try:
        import nvidia
    except ImportError:
        pass
    else:
        # The wheels install the toolkit as a versioned subpackage; prefer the
        # newest if several are present.
        for root in map(pathlib.Path, nvidia.__path__):
            for cand in sorted(root.glob("cu[0-9]*"), reverse=True):
                if (cand / "bin" / "nvcc").exists():
                    return cand

    nvcc = shutil.which("nvcc")
    if nvcc:
        return pathlib.Path(nvcc).resolve().parent.parent

    raise RuntimeError(
        "No CUDA toolkit found. Either activate this repo's venv (`uv sync` "
        "installs nvidia-cuda-nvcc), or point $CUDA_HOME at a system toolkit."
    )


@functools.cache
def cuda_lib_dir() -> pathlib.Path:
    """Directory holding the toolkit's libraries: lib/ on pip, lib64/ on system."""
    root = cuda_home()
    for name in ("lib64", "lib"):
        d = root / name
        if d.is_dir() and any(d.glob("libcudart*")):
            return d
    raise RuntimeError(f"no CUDA library directory under {root}")


def nvcc() -> pathlib.Path:
    return cuda_home() / "bin" / "nvcc"


@functools.cache
def link_shim_dir() -> pathlib.Path:
    """A -L directory of unversioned .so symlinks, created on demand.

    The pip CUDA wheels ship only `libcudart.so.13`; `ld -lcudart` insists on
    a bare `libcudart.so`. Rather than committing a symlink with a machine
    specific target, we point one at whatever this environment resolved to.
    """
    shim = BUILD_DIR / "lib_shim"
    shim.mkdir(parents=True, exist_ok=True)
    for stem in ("libcudart", "libcuda"):
        link = shim / f"{stem}.so"
        versioned = sorted(cuda_lib_dir().glob(f"{stem}.so.*"))
        if not versioned:
            continue
        target = versioned[-1]
        if link.is_symlink() and link.readlink() == target:
            continue
        link.unlink(missing_ok=True)
        link.symlink_to(target)
    return shim


@functools.cache
def gpu_arch(default: str = DEFAULT_ARCH) -> str:
    """`sm_XX` for the GPU in this machine.

    Asks nvidia-smi first: it costs milliseconds, where importing torch just
    to read a compute capability costs seconds (and the Makefile calls this).
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.split("\n")[0].strip()
        major, minor = out.split(".")
        return f"sm_{int(major)}{int(minor)}"
    except Exception:
        pass

    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        return f"sm_{major}{minor}"

    return default


def activate() -> pathlib.Path:
    """Export CUDA_HOME and put nvcc on $PATH for tools that shell out to it.

    torch.utils.cpp_extension finds nvcc this way. Returns the toolkit root.
    """
    root = cuda_home()
    os.environ.setdefault("CUDA_HOME", str(root))
    bindir = str(root / "bin")
    if bindir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    return root


def main() -> int:
    what = sys.argv[1] if len(sys.argv) > 1 else "home"
    printers = {
        "home": lambda: cuda_home(),
        "lib": lambda: cuda_lib_dir(),
        "nvcc": lambda: nvcc(),
        "shim": lambda: link_shim_dir(),
        "arch": lambda: gpu_arch(),
    }
    if what not in printers:
        print(f"usage: {sys.argv[0]} [{'|'.join(printers)}]", file=sys.stderr)
        return 2
    print(printers[what]())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
