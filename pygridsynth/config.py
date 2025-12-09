from dataclasses import dataclass, field

from .loop_controller import LoopController


@dataclass
class GridsynthConfig:
    dps: int | None = None
    seed: int = 0
    dloop: int = 10
    floop: int = 10
    dtimeout: float | None = None
    ftimeout: float | None = None
    verbose: int = 0
    measure_time: bool = False
    show_graph: bool = False
    up_to_phase: bool = False

    loop_controller: LoopController = field(init=False)

    def __post_init__(self) -> None:
        self.loop_controller = LoopController(
            dloop=self.dloop,
            floop=self.floop,
            dtimeout=self.dtimeout,
            ftimeout=self.ftimeout,
        )
