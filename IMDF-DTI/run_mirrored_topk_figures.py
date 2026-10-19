"""Plot independently ranked feature-mask indices in paired drug/target views."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drugbank-weights', default='DrugBank/ensemble_weights.npy')
    parser.add_argument('--biosnap-weights', default='BIOSNAP/ensemble_weights.npy')
    parser.add_argument('--output-dir', default='figures/topk')
    parser.add_argument('--top-k', type=int, default=15)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    from utils.Visualization_mirrored_topk import generate_mirrored_topk_figures_from_weights
    for name, path in [('DrugBank', args.drugbank_weights), ('BioSNAP', args.biosnap_weights)]:
        if not Path(path).is_file():
            parser.error(f'Missing trusted, locally generated weight file: {path}')
        generate_mirrored_topk_figures_from_weights(
            weight_path=path, save_dir=str(Path(args.output_dir) / name),
            dataset_name=name, top_k=args.top_k, display_mode='minmax')


if __name__ == '__main__':
    main()
