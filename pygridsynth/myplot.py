from typing import Iterable

from .region import Ellipse, Rectangle
from .ring import DOmega


def plot_sol(
    sol_list: list[Iterable[DOmega]],
    ellipseA: Ellipse,
    ellipseB: Ellipse,
    bboxA: Rectangle | None = None,
    bboxB: Rectangle | None = None,
    color_list: list[str] | None = None,
    size_list: list[float] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    if color_list is None:
        color_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if size_list is None:
        size_list = [5] * len(sol_list)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_aspect("equal")
    ax1.set_xlabel(r"$\mathrm{Re}[u]$")
    ax1.set_ylabel(r"$\mathrm{Im}[u]$")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_aspect("equal")
    ax2.set_xlabel(r"$\mathrm{Re}[u^\bullet]$")
    ax2.set_ylabel(r"$\mathrm{Im}[u^\bullet]$")

    for sol, color, size in zip(sol_list, color_list, size_list):
        x = [u.real for u in sol]
        y = [u.imag for u in sol]
        ax1.scatter(x, y, c=color, s=size)
        x = [u.conj_sq2.real for u in sol]
        y = [u.conj_sq2.imag for u in sol]
        ax2.scatter(x, y, c=color, s=size)

    ellipseA.plot(ax1)
    ellipseB.plot(ax2)

    if bboxA is not None:
        bboxA.plot(ax1)
    if bboxB is not None:
        bboxB.plot(ax2)

    plt.show()
