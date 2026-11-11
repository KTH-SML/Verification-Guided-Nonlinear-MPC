"""
Uniform console logging for all processes.

Every line has the form
    [t =   2.40 s] Verifier   | message
where t is the simulation time (blank for setup messages), so lines printed by
different processes can be read in order.
"""


def log(source: str, msg: str, t: float | None = None) -> None:
    stamp = f"t = {t:6.2f} s" if t is not None else " " * 12
    print(f"[{stamp}] {source:<10} | {msg}", flush=True)


def rule(title: str = "", width: int = 78) -> None:
    """Section separator, e.g. for summaries."""
    if title:
        title = f" {title} "
    print(f"{title:=^{width}}", flush=True)
