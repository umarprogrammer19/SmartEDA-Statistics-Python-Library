import pandas as pd
import sys
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from src.eda.full_eda import full_eda

console = Console()


def main():
    # Display welcome header
    console.print(Panel("[bold blue]SMART EDA LIBRARY[/bold blue]", expand=False))
    console.print("[bold green]AUTO EDA STARTED[/bold green]\n")

    # 1. Load your dataset here
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Loading dataset...", total=None)
            df = pd.read_csv("data.csv")

        console.print("[bold green][OK][/bold green] Dataset loaded successfully.", style="green")
    except FileNotFoundError:
        console.print("[bold red][ERROR][/bold red] [red]data.csv not found.[/red]")
        console.print("[yellow]Please place your dataset in the same folder as main.py[/yellow]")
        return

    # 2. Show dataset info and ask for target column
    console.print("\n[bold]Columns in your dataset:[/bold]")

    # Create a table to display columns nicely
    table = Table(title="Dataset Columns", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Column Name", style="cyan")
    table.add_column("Data Type", justify="right")

    for i, col in enumerate(df.columns, 1):
        table.add_row(str(i), col, str(df[col].dtype))

    console.print(table)

    # Check for visualization flag in command line arguments
    generate_viz = "--viz" in sys.argv or "-v" in sys.argv

    # Get target column from command line argument or user input
    if len(sys.argv) > 1:
        # Remove viz flag from sys.argv if present to get the actual target
        args = [arg for arg in sys.argv[1:] if arg not in ["--viz", "-v"]]
        if args:
            target = args[0]
            console.print(f"\n[bold blue]Using target column from command line:[/bold blue] [italic]{target}[/italic]")
        else:
            # Interactive selection
            console.print("\n[bold]Choose target column:[/bold]")
            for i, col in enumerate(df.columns, 1):
                console.print(f"[{i}] {col}")

            while True:
                try:
                    choice = Prompt.ask("\nEnter column number or name", default="")
                    if choice.isdigit():
                        idx = int(choice) - 1
                        if 0 <= idx < len(df.columns):
                            target = df.columns[idx]
                            break
                        else:
                            console.print("[red]Invalid column number. Please try again.[/red]")
                    else:
                        target = choice.strip()
                        if target in df.columns:
                            break
                        else:
                            console.print(f"[red]'{target}' does not exist in dataset. Please try again.[/red]")
                except KeyboardInterrupt:
                    console.print("\n[yellow]Operation cancelled by user.[/yellow]")
                    return
    else:
        # Interactive selection
        console.print("\n[bold]Choose target column:[/bold]")
        for i, col in enumerate(df.columns, 1):
            console.print(f"[{i}] {col}")

        while True:
            try:
                choice = Prompt.ask("\nEnter column number or name", default="")
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(df.columns):
                        target = df.columns[idx]
                        break
                    else:
                        console.print("[red]Invalid column number. Please try again.[/red]")
                else:
                    target = choice.strip()
                    if target in df.columns:
                        break
                    else:
                        console.print(f"[red]'{target}' does not exist in dataset. Please try again.[/red]")
            except KeyboardInterrupt:
                console.print("\n[yellow]Operation cancelled by user.[/yellow]")
                return

    if target not in df.columns:
        console.print(f"[red]ERROR: '{target}' does not exist in dataset.[/red]")
        return

    # Determine if visualizations should be generated
    cmd_generate_viz = "--viz" in sys.argv or "-v" in sys.argv
    if not cmd_generate_viz:
        # Ask if user wants visualizations
        viz_choice = Prompt.ask("\n[bold]Would you like to generate visualizations?[/bold] ([blue]y[/blue]/[red]n[/red])", default="n")
        cmd_generate_viz = viz_choice.lower() in ['y', 'yes', 'true', '1']

    # Run automated EDA with progress bar
    console.print(f"\n[bold blue]Running full EDA on target:[/bold blue] [italic]{target}[/italic]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Performing EDA analysis...", total=None)
        clean_df, insights = full_eda(df, target=target, generate_viz=cmd_generate_viz)

    # 4. Display Insights in a nice format
    console.print("\n[bold green]EDA Insights Summary:[/bold green]\n")

    for key, value in insights.items():
        panel_title = f"[bold]{key.upper()}[/bold]"
        if isinstance(value, dict):
            # Format dictionary nicely
            formatted_value = "\n".join([f"{k}: {v}" for k, v in value.items()])
            console.print(Panel(formatted_value, title=panel_title, border_style="blue"))
        elif hasattr(value, '__len__') and len(value) > 0:
            # Format pandas Series/DataFrame
            console.print(Panel(str(value), title=panel_title, border_style="blue"))
        else:
            # Plain text for other types
            console.print(Panel(str(value), title=panel_title, border_style="blue"))

    # 5. Save cleaned dataset
    clean_df.to_csv("cleaned_output.csv", index=False)
    console.print("\n[bold green][OK] Cleaned dataset saved as:[/bold green] [italic]cleaned_output.csv[/italic]")

    console.print("\n[bold green][COMPLETE] EDA Complete![/bold green]\n")


if __name__ == "__main__":
    main()
