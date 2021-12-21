"""
Collect information from all BitVectors_Hist.txt files in a directory.
"""

import sys

from rouls.dreem_utils import read_all_bitvector_hist_files

em_clustering_dir = sys.argv[1]
output_file = sys.argv[2]
bv_hist_info = read_all_bitvector_hist_files(em_clustering_dir, missing="ignore")
bv_hist_info.to_csv(output_file, sep="\t")
print(bv_hist_info)

