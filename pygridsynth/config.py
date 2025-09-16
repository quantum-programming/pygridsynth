from dataclasses import dataclass, field
from typing import Optional

from .loop_controller import LoopController


@dataclass
class GridsynthConfig:
    dps: Optional[int] = None
    seed: int = 0
    dloop: int = 10
    floop: int = 10
    dtimeout: Optional[float] = None
    ftimeout: Optional[float] = None
    verbose: bool = False
    measure_time: bool = False
    show_graph: bool = False
    upto_phase: bool = False

    loop_controller: LoopController = field(init=False)

    def __post_init__(self):
        self.loop_controller = LoopController(
            dloop=self.dloop,
            floop=self.floop,
            dtimeout=self.dtimeout,
            ftimeout=self.ftimeout,
        )
