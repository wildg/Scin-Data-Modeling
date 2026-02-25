"""Console script for scin_data_modeling."""

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

app = typer.Typer(
    name="scin_data_modeling",
    help="SCIN data pipeline: download raw data and build cleaned datasets.",
    no_args_is_help=True,
)
console = Console()


# ── internal helpers (shared by individual subcommands and the pipeline) ───────


def _download(raw_dir: Path, images: bool) -> None:
    from scin_data_modeling.data.download import download_csvs, download_images

    console.print(Rule("[bold blue]Step 1 — Download[/bold blue]"))
    download_csvs(local_dir=raw_dir)
    if images:
        download_images(local_dir=raw_dir)
    else:
        console.print("[yellow]Skipping images (--no-images)[/yellow]")
    console.print("[green]✓ Download complete.[/green]")


def _preprocess(
    raw_dir: Path,
    out_dir: Path,
    test_size: float,
    seed: int,
    create_split: bool,
    create_embeddings: bool,
    embedding_backbone: str,
    embedding_device: str,
    embedding_split: str,
) -> None:
    from scin_data_modeling.data.preprocess import (
        build_clean_df,
        create_train_test_split,
        load_combined_df,
        save_clean_data,
        save_splits,
    )

    console.print(Rule("[bold blue]Step 2 — Preprocess[/bold blue]"))

    console.print("Loading CSVs…")
    combined_df = load_combined_df(raw_dir=raw_dir)
    console.print(f"  Loaded [cyan]{len(combined_df):,}[/cyan] cases.")

    console.print("Cleaning merged dataset…")
    cleaned_df = build_clean_df(combined_df)
    dropped = len(combined_df) - len(cleaned_df)
    console.print(
        f"  Retained [cyan]{len(cleaned_df):,}[/cyan] cleaned cases "
        f"(dropped {dropped:,} with missing labels/images)."
    )

    import json as _json
    from collections import Counter

    all_conditions: Counter[str] = Counter()
    for label_json in cleaned_df["label"]:
        for cond in _json.loads(label_json):
            all_conditions[cond] += 1
    console.print(f"  Unique skin conditions (from top-3 label): [cyan]{len(all_conditions):,}[/cyan]")

    out_path = save_clean_data(cleaned_df, out_dir=out_dir)
    console.print(f"  Cleaned dataset: [cyan]{out_path}[/cyan]")

    if create_embeddings and not create_split:
        console.print("[yellow]Embeddings require split files; enabling --create-split automatically.[/yellow]")
        create_split = True

    if create_split:
        console.print(f"Splitting {1 - test_size:.0%} train / {test_size:.0%} test (seed={seed})…")
        train_df, test_df = create_train_test_split(cleaned_df, test_size=test_size, random_state=seed)
        save_splits(train_df, test_df, out_dir=out_dir)
    else:
        console.print("[dim]Skipping train/test split (--no-create-split).[/dim]")

    if create_embeddings:
        _embed(
            backbone_name=embedding_backbone,
            split=embedding_split,
            device_str=embedding_device,
            processed_dir=out_dir,
        )
    else:
        console.print("[dim]Skipping embeddings (--no-create-embeddings).[/dim]")

    console.print("[green]✓ Preprocessing complete.[/green]")


def _embed(backbone_name: str, split: str, device_str: str, processed_dir: Path) -> None:
    import torch

    from scin_data_modeling.data.embed import embed_split
    from scin_data_modeling.models.backbone import get_backbone, list_backbones

    console.print(Rule("[bold blue]Step 3 — Embed[/bold blue]"))
    console.print(f"  Backbone:  [cyan]{backbone_name}[/cyan]")
    console.print(f"  Split:     [cyan]{split}[/cyan]")
    console.print(f"  Device:    [cyan]{device_str}[/cyan]")
    console.print(f"  Available: {', '.join(list_backbones())}")

    device = torch.device(device_str)
    model, transform, embed_dim = get_backbone(backbone_name, pretrained=True)
    console.print(f"  Embed dim: [cyan]{embed_dim}[/cyan]")

    splits = ["train", "test"] if split == "both" else [split]
    for s in splits:
        console.print(f"\nEmbedding [bold]{s}[/bold] split…")
        out_path = embed_split(
            split=s,
            backbone_name=backbone_name,
            model=model,
            transform=transform,
            device=device,
            processed_dir=processed_dir,
        )
        console.print(f"  → {out_path}")

    console.print("[green]✓ Embedding complete.[/green]")


def _train(mode: str, processed_dir: Path, model_dir: Path, backbone_name: str, device_str: str) -> None:
    console.print(Rule("[bold blue]Step 4 — Train[/bold blue]"))

    if mode == "frozen":
        train_emb = processed_dir / "embeddings_train.npz"
        if not train_emb.exists():
            console.print(
                f"[red]Embeddings not found at {train_emb}.[/red]\n"
                "  Run [bold]scin_data_modeling embed[/bold] first."
            )
            raise typer.Exit(code=1)
        console.print(f"  Mode: [cyan]frozen[/cyan] — training head on cached embeddings")
        console.print(f"  Embeddings: {train_emb}")
        console.print(
            "[yellow]Head training not yet implemented. "
            "Add training loop to [bold]scin_data_modeling/models/[/bold][/yellow]"
        )
    elif mode == "finetune":
        console.print(f"  Mode: [cyan]finetune[/cyan] — end-to-end with GCS streaming")
        console.print(f"  Backbone: [cyan]{backbone_name}[/cyan]  Device: [cyan]{device_str}[/cyan]")
        console.print(
            "[yellow]Fine-tune training not yet implemented. "
            "Add training loop to [bold]scin_data_modeling/models/[/bold][/yellow]"
        )
    else:
        console.print(f"[red]Unknown mode: {mode!r}. Use 'frozen' or 'finetune'.[/red]")
        raise typer.Exit(code=1)


def _evaluate(processed_dir: Path, model_dir: Path) -> None:
    console.print(Rule("[bold blue]Step 5 — Evaluate[/bold blue]"))
    console.print(
        "[yellow]Evaluation not yet implemented. "
        "Add metric code to [bold]scin_data_modeling/evaluation/[/bold][/yellow]"
    )


# ── subcommands ────────────────────────────────────────────────────────────────


@app.command()
def download(
    raw_dir: Path = typer.Option(Path("data/raw"), help="Directory to save raw data."),
    images: bool = typer.Option(False, "--images/--no-images", help="Also download images (several GB)."),
) -> None:
    """Download the SCIN dataset CSVs and (optionally) images from GCS."""
    _download(raw_dir=raw_dir, images=images)


@app.command()
def preprocess(
    raw_dir: Path = typer.Option(Path("data/raw"), help="Directory containing raw CSVs."),
    out_dir: Path = typer.Option(Path("data/processed"), help="Output directory for cleaned data."),
    test_size: float = typer.Option(0.2, help="Fraction of data reserved for the test split."),
    seed: int = typer.Option(42, help="Random seed for the train/test split."),
    create_split: bool = typer.Option(
        False,
        "--create-split/--no-create-split",
        help="Also create train/test CSV files from cleaned data.",
    ),
    create_embeddings: bool = typer.Option(
        False,
        "--create-embeddings/--no-create-embeddings",
        help="Also generate image embeddings using a frozen CNN backbone.",
    ),
    embedding_backbone: str = typer.Option(
        "resnet50",
        help="Backbone model for embeddings (resnet50, efficientnet_b0).",
    ),
    embedding_device: str = typer.Option(
        "cpu",
        help="Torch device for embeddings (cpu, cuda, mps).",
    ),
    embedding_split: str = typer.Option(
        "both",
        help="Which split to embed when creating embeddings: 'train', 'test', or 'both'.",
    ),
) -> None:
    """Merge and clean SCIN raw data; optionally split and create image embeddings."""
    _preprocess(
        raw_dir=raw_dir,
        out_dir=out_dir,
        test_size=test_size,
        seed=seed,
        create_split=create_split,
        create_embeddings=create_embeddings,
        embedding_backbone=embedding_backbone,
        embedding_device=embedding_device,
        embedding_split=embedding_split,
    )


@app.command()
def embed(
    backbone: str = typer.Option("resnet50", help="Backbone model name (resnet50, efficientnet_b0)."),
    split: str = typer.Option("both", help="Which split to embed: 'train', 'test', or 'both'."),
    device: str = typer.Option("cpu", help="Torch device (cpu, cuda, mps)."),
    processed_dir: Path = typer.Option(Path("data/processed"), help="Directory with train.csv / test.csv."),
) -> None:
    """Stream images from GCS, encode with a frozen backbone, and save embeddings."""
    _embed(backbone_name=backbone, split=split, device_str=device, processed_dir=processed_dir)


@app.command()
def train(
    mode: str = typer.Option("frozen", help="Training mode: 'frozen' (head only) or 'finetune' (end-to-end)."),
    backbone: str = typer.Option("resnet50", help="Backbone for finetune mode."),
    device: str = typer.Option("cpu", help="Torch device (cpu, cuda, mps)."),
    processed_dir: Path = typer.Option(Path("data/processed"), help="Directory containing processed splits."),
    model_dir: Path = typer.Option(Path("models"), help="Directory to save trained model artefacts."),
) -> None:
    """Train a model. Use --mode frozen (on cached embeddings) or --mode finetune (stream from GCS)."""
    _train(mode=mode, processed_dir=processed_dir, model_dir=model_dir, backbone_name=backbone, device_str=device)


@app.command()
def evaluate(
    processed_dir: Path = typer.Option(Path("data/processed"), help="Directory containing processed splits."),
    model_dir: Path = typer.Option(Path("models"), help="Directory containing trained model artefacts."),
) -> None:
    """Evaluate the trained model on the test split. (Not yet implemented.)"""
    _evaluate(processed_dir=processed_dir, model_dir=model_dir)


@app.command()
def pipeline(
    raw_dir: Path = typer.Option(Path("data/raw"), help="Raw data directory."),
    processed_dir: Path = typer.Option(Path("data/processed"), help="Processed data directory."),
    test_size: float = typer.Option(0.2, help="Fraction of data reserved for the test split."),
    seed: int = typer.Option(42, help="Random seed for the train/test split."),
    images: bool = typer.Option(False, "--images/--no-images", help="Download images during the download step."),
    create_split: bool = typer.Option(
        False,
        "--create-split/--no-create-split",
        help="Also create train/test CSV files from cleaned data.",
    ),
    create_embeddings: bool = typer.Option(
        False,
        "--create-embeddings/--no-create-embeddings",
        help="Also generate image embeddings using a frozen CNN backbone.",
    ),
    embedding_backbone: str = typer.Option(
        "resnet50",
        help="Backbone model for embeddings (resnet50, efficientnet_b0).",
    ),
    embedding_device: str = typer.Option(
        "cpu",
        help="Torch device for embeddings (cpu, cuda, mps).",
    ),
    embedding_split: str = typer.Option(
        "both",
        help="Which split to embed when creating embeddings: 'train', 'test', or 'both'.",
    ),
    skip_download: bool = typer.Option(False, "--skip-download/--no-skip-download", help="Skip the download step."),
    skip_preprocess: bool = typer.Option(
        False, "--skip-preprocess/--no-skip-preprocess", help="Skip the preprocess step."
    ),
) -> None:
    """Run the base data pipeline: download → preprocess (optional embeddings)."""
    console.print(
        Panel(
            "[bold cyan]SCIN Base Data Pipeline[/bold cyan]",
            subtitle="download → preprocess",
        )
    )

    if not skip_download:
        if create_embeddings and not images:
            console.print("[dim]Embeddings stream images from GCS; local image download is not required.[/dim]")
        _download(raw_dir=raw_dir, images=images)
    else:
        console.print("[dim]Skipping download.[/dim]")

    if not skip_preprocess:
        _preprocess(
            raw_dir=raw_dir,
            out_dir=processed_dir,
            test_size=test_size,
            seed=seed,
            create_split=create_split,
            create_embeddings=create_embeddings,
            embedding_backbone=embedding_backbone,
            embedding_device=embedding_device,
            embedding_split=embedding_split,
        )
    else:
        console.print("[dim]Skipping preprocess.[/dim]")

    console.rule("[bold green]Pipeline complete[/bold green]")


if __name__ == "__main__":
    app()
