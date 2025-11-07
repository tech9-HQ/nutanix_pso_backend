import time
from contextlib import contextmanager

@contextmanager
def timeit(label: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000
        print(f"[timeit] {label}: {dt:.1f} ms")
