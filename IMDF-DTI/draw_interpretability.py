"""Plot learned mask profiles and within-view weights, not biological attributions."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drugbank-weights', default='DrugBank/ensemble_weights.npy')
    parser.add_argument('--biosnap-weights', default='BIOSNAP/ensemble_weights.npy')
    parser.add_argument('--output-dir', default='figures/profiles')
    parser.add_argument('--top-k', type=int, default=15)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    from utils.Visualization_paper_style_gradient_lollipop import (
        generate_interpretability_figures_from_weights, generate_dfgu_compare_figure)
    for name, path in [('DrugBank', args.drugbank_weights), ('BioSNAP', args.biosnap_weights)]:
        if not Path(path).is_file():
            parser.error(f'Missing trusted, locally generated weight file: {path}')
        generate_interpretability_figures_from_weights(
            weight_path=path, save_dir=str(Path(args.output_dir) / name),
            display_mode='minmax', cmap='YlGnBu', top_k=args.top_k)
    generate_dfgu_compare_figure(args.drugbank_weights, args.biosnap_weights,
                                 str(Path(args.output_dir) / 'DFGU_compare'))


if __name__ == '__main__':
    main()
