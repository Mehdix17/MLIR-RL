import multiprocessing
from multiprocessing import Process, Queue
from queue import Empty as QueueEmpty
from typing import Callable, Optional, TypeVar

T = TypeVar('T')
ENABLED = False
T = TypeVar('T')


class BindingsProcess:
    @staticmethod
    def call(func: Callable[..., T], *args, timeout: Optional[float] = None) -> T:
        if not ENABLED:
            return func(*args)

        def func_wrapper(q: Queue):
            try:
                q.put(func(*args))
            except Exception as e:
                q.put(e)

        q = Queue()
        p = Process(target=func_wrapper, args=(q,), daemon=True)
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.kill()
            p.join()
            raise TimeoutError(f"Bindings call {func.__name__} timed out after {timeout}s")

        ec = p.exitcode
        if ec:
            raise Exception(f"Bindings call {func.__name__} failed with exit code: {ec}")

        try:
            res = q.get_nowait()
        except QueueEmpty:
            func_name = getattr(func, '__name__', str(func))
            raise Exception(
                f"Bindings call {func_name} failed: subprocess exited without result "
                "(likely a crash in native code)"
            )

        if isinstance(res, Exception):
            raise res
        return res
