"""ShopTheLook source package.

macOS note: torch and faiss-cpu wheels each bundle libomp. Run
`python scripts/fix_macos_openmp.py` once after installing requirements so both
share a single OpenMP runtime (otherwise the process can abort or segfault).
"""
