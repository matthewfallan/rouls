"""
Compute data-structure correlation (DSC) using one of several metrics.
"""


import argparse

import pandas as pd

from dreem_utils import get_data_structure_agreement, read_mutation_rates
from struct_utils import read_structure_file


def compute_correlation(args):
    start_pos = args.start_pos
    if start_pos is None:
        # If the start position is not given, set it to the first base in the
        # mutation rate file.
        mus = read_mutation_rates(args.mutation_file, include_gu=True)
        if len(mus.index) > 0:
            start_pos = mus.index.tolist()[0]
        else:
            raise ValueError("mutation rates is empty")
    paired, seq = read_structure_file(args.structure_file, start_pos=start_pos,
            title_mode="number")
    mus = read_mutation_rates(args.mutation_file, flatten=True,
                include_gu=args.include_gu, seq=seq, start_pos=start_pos)
    if isinstance(mus, pd.DataFrame):
        if args.cluster is None:
            raise ValueError("Cluster must be specified if file has >1 cluster.")
        else:
            mus = mus[args.cluster]
    dsas = dict()
    for structure in paired.columns:
        dsa = get_data_structure_agreement(args.metric, paired[structure], mus,
                    min_paired=args.min_paired, min_unpaired=args.min_unpaired)
        dsas[structure] = dsa
    dsas = pd.Series(dsas)
    return dsas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("structure_file",
            help="file of the RNA structure in CT or dot-bracket format")
    parser.add_argument("mutation_file",
            help="file of the mutation rates in Clusters_Mu.txt format")
    parser.add_argument("metric", help="correlation metric; can be 'RBC' for"
            " rank-biserial correlation or 'CLES' for common language effect"
            " size")
    parser.add_argument("--cluster", nargs="?")
    parser.add_argument("--include_gu", action="store_true")
    parser.add_argument("--min_paired", nargs="?", type=int, default=10)
    parser.add_argument("--min_unpaired", nargs="?", type=int, default=10)
    parser.add_argument("--start_pos", nargs="?", type=int, default=None)
    parser.add_argument("--output_file", nargs="?", type=int, default=None)
    args = parser.parse_args()
    correlations = compute_correlation(args)
    print(correlations)
    if args.output_file:
        if os.path.exists(args.output_file):
            raise ValueError(f"output file already exists: {args.output_file}")
        fname, ext = os.path.splitext(args.output_file)
        if ext == ".csv":
            correlations.to_csv(args.output_file, sep=",")
        elif ext in [".tab", ".txt", ""]:
            correlations.to_csv(args.output_file, sep="\t")
        elif ext in [".xls", ".xlsx"]:
            correlations.to_excel(args.output_file)
        else:
            raise ValueError(f"unrecognized extension: {ext}")

