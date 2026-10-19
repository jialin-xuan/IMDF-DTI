"""Command-line entry point for IMDF-DTI and probability-averaged IMDF-DTI-B."""

import argparse
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        prog="IMDF-DTI",
        description="Masked multimodal drug-target interaction prediction.",
        epilog="Run from the repository root. Defaults are documented in REPRODUCIBILITY.md.",
    )
    parser.add_argument("dataSetName", choices=["DrugBank", "Davis", "BIOSNAP", "BD2D"])
    parser.add_argument("-m", "--model", default="IMDF-DTI",
                        choices=["IMDF-DTI", "IMDF-DTI-B", "MIF-DTI", "MIF-DTI-B"])
    parser.add_argument("-s", "--seed", type=int, default=114514)
    parser.add_argument("-f", "--fold", type=int, default=5)
    parser.add_argument("-g", "--gpu", type=int, default=0)
    parser.add_argument("--ablation", default="full", choices=[
        "full", "wo_XMask", "wo_DFGU", "wo_RESCAL", "wo_staged_opt"])
    parser.add_argument("--output-dir", help="Training/evaluation output directory.")
    parser.add_argument("--checkpoint-dir", help="Ensemble input directory containing fold subdirectories.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.fold < 2:
        parser.error("--fold must be at least 2")
    dataset_file = Path("DataSets") / f"{args.dataSetName}.txt"
    if not dataset_file.is_file():
        parser.error(f"Missing {dataset_file}; see DataSets/README.md")

    model_name = args.model.replace("MIF-DTI", "IMDF-DTI", 1) if args.model.startswith("MIF-") else args.model
    if model_name.endswith("-B"):
        if args.dataSetName == "BD2D" or args.ablation != "full":
            parser.error("The probability-averaging entry point supports the three full-model benchmarks.")
        from ensemble_final import run_ensemble
        run_ensemble(dataset=args.dataSetName, seed=args.seed, gpu=args.gpu,
                     folds=args.fold, checkpoint_dir=args.checkpoint_dir,
                     output_dir=args.output_dir)
        return

    if args.checkpoint_dir:
        parser.error("--checkpoint-dir is an ensemble option, not a training resume option")
    output_dir = Path(args.output_dir or (
        args.dataSetName if args.ablation == "full" else f"{args.dataSetName}_{args.ablation}"))
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"Output directory {output_dir} is not empty; use a new --output-dir to protect earlier runs.")

    import torch
    from RunModel import run_IMDF_model
    from model import IMDFDTI
    if torch.cuda.is_available() and not 0 <= args.gpu < torch.cuda.device_count():
        parser.error("--gpu is outside the visible CUDA device range")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Source-snapshot training: CE loss; see REPRODUCIBILITY.md before claiming paper reproduction.")
    run_IMDF_model(SEED=args.seed, DATASET=args.dataSetName, MODEL=IMDFDTI,
                   K_Fold=args.fold, LOSS="CELoss", device=device,
                   ABLATION=args.ablation, OUTPUT_DIR=str(output_dir))


if __name__ == "__main__":
    main()
