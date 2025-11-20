from typing import Optional
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn

console = Console()


def get_progress() -> Progress:
    return Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def info(msg: str, style: Optional[str] = None) -> None:
    console.print(msg, style=style)
