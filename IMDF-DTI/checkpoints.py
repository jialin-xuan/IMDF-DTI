"""Resolve existing state-dict checkpoints without depending on a saved GPU name."""

from pathlib import Path


def find_checkpoint(fold_dir):
    folder = Path(fold_dir)
    matches = sorted(folder.glob("valid_best_checkpoint-*.pth"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one valid_best_checkpoint-*.pth in {folder}; found {len(matches)}. "
            "Select the intended run explicitly; checkpoints are not chosen by test performance."
        )
    return matches[0]
