"""Console script for scin_data_modeling."""

import typer
from rich.console import Console

from scin_data_modeling import utils

app = typer.Typer()
console = Console()


@app.command()
def main() -> None:
    """Console script for scin_data_modeling."""
    console.print("Replace this message by putting your code into "
               "scin_data_modeling.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    utils.do_something_useful()


if __name__ == "__main__":
    app()
