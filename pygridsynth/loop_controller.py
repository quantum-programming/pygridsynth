import math
import time
from typing import Optional


class LoopController:
    def __init__(
        self,
        dloop: int = 10,
        floop: int = 10,
        dtimeout: Optional[float] = None,
        ftimeout: Optional[float] = None,
    ) -> None:
        self.diophantine_loops: int = dloop
        self.factoring_loops: int = floop
        self.diophantine_timeout: float = (
            dtimeout / 1000 if dtimeout is not None else float("inf")
        )
        self.factoring_timeout: float = (
            ftimeout / 1000 if ftimeout is not None else float("inf")
        )

        self._d_counter: int = 0
        self._f_counter: int = 0
        self._d_start_time: float = float("nan")
        self._f_start_time: float = float("nan")

    def start_diophantine(self) -> None:
        self._d_counter = 0
        self._d_start_time = time.time()

    def start_factoring(self) -> None:
        self._f_counter = 0
        self._f_start_time = time.time()

    def check_diophantine_continue(self) -> bool:
        if self._d_counter >= self.diophantine_loops:
            return False

        if not math.isinf(self.diophantine_timeout):
            assert not math.isnan(self._d_start_time)
            elapsed = time.time() - self._d_start_time
            if elapsed >= self.diophantine_timeout:
                return False

        self._d_counter += 1
        return True

    def check_factoring_continue(self) -> bool:
        if self._f_counter >= self.factoring_loops:
            return False

        if not math.isinf(self.factoring_timeout):
            assert not math.isnan(self._f_start_time)
            elapsed = time.time() - self._f_start_time
            if elapsed >= self.factoring_timeout:
                return False

        self._f_counter += 1
        return True

    @property
    def diophantine_iteration_count(self) -> int:
        return self._d_counter

    @property
    def factoring_iteration_count(self) -> int:
        return self._f_counter

    def reset_counters(self) -> None:
        self._d_counter = 0
        self._f_counter = 0
        self._d_start_time = float("nan")
        self._f_start_time = float("nan")
