"""macOS only: make faiss-cpu and torch share ONE OpenMP runtime.

The PyPI wheels of torch and faiss-cpu each bundle their own libomp.dylib.
When both load into one process (our pipeline needs both), two OpenMP runtimes
run side by side. That either aborts ("OMP: Error #15") or segfaults at random
(we saw exit code 139 while embedding).

Fix: replace faiss's bundled copy with a symlink to torch's. dyld resolves the
symlink to the same file, so only one runtime is loaded. Idempotent. The
original file is kept as libomp.dylib.orig.

Run once after `pip install -r requirements.txt`:
    python scripts/fix_macos_openmp.py
"""
import importlib.util
import os
import sys
from pathlib import Path


def site_dir(pkg: str) -> Path:
    spec = importlib.util.find_spec(pkg)
    return Path(spec.origin).parent


def main() -> None:
    if sys.platform != "darwin":
        print("Not macOS: nothing to do.")
        return
    torch_omp = site_dir("torch") / "lib" / "libomp.dylib"
    faiss_omp = site_dir("faiss") / ".dylibs" / "libomp.dylib"
    if not torch_omp.exists() or not faiss_omp.exists() and not faiss_omp.is_symlink():
        print(f"Expected files not found ({torch_omp}, {faiss_omp}); nothing changed.")
        return
    if faiss_omp.is_symlink() and faiss_omp.resolve() == torch_omp.resolve():
        print("Already fixed: faiss uses torch's libomp.")
        return
    backup = faiss_omp.with_suffix(".dylib.orig")
    if not backup.exists():
        faiss_omp.rename(backup)
    else:
        faiss_omp.unlink()
    os.symlink(torch_omp, faiss_omp)
    print(f"Linked {faiss_omp} -> {torch_omp}")


if __name__ == "__main__":
    main()
