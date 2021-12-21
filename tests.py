"""
ROULS - Unit test module

Purpose: Unit tests for all functions.
Author: Matty Allan
Modified: 2021-05-23
"""


import logging
logging.basicConfig(filename="tests.log", filemode="w", level=logging.DEBUG)
import os
import unittest as ut

import numpy as np
import pandas as pd
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

import seq_utils
import struct_utils
import dreem_utils


data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
assert os.path.isdir(data_dir)


SEQS = {
        "RRE": "GGAGCTTTGTTCCTTGGGTTCTTGGGAGCAGCAGGAAGCACTATGGGCGCAGCGTCAATGACG"
               "CTGACGGTACAGGCCAGACAATTATTGTCTGATATAGTGCAGCAGCAGAACAATTTGCTGAGG"
               "GCTATTGAGGCGCAACAGCATCTGTTGCAACTCACAGTCTGGGGCATCAAACAGCTCCAGGCA"
               "AGAATCCTGGCTGTGGAAAGATACCTAAAGGATCAACAGCTCC"
       }

AC_POS = {
        "RRE": [3, 5, 12, 13, 21, 27, 29, 30, 32, 33, 36, 37, 39, 40, 41, 43,
                48, 50, 51, 53, 56, 57, 58, 61, 62, 64, 67, 68, 72, 73, 74, 77,
                78, 79, 81, 82, 83, 84, 87, 92, 95, 97, 99, 103, 104, 106, 107,
                109, 110, 112, 113, 114, 115, 116, 121, 124, 128, 130, 134, 137,
                139, 140, 141, 142, 143, 145, 146, 148, 154, 155, 156, 157, 159,
                160, 161, 162, 165, 171, 172, 174, 175, 176, 177, 178, 179, 181,
                183, 184, 185, 188, 189, 190, 192, 193, 195, 196, 200, 206, 207,
                208, 210, 212, 213, 214, 216, 217, 218, 221, 223, 224, 225, 226,
                227, 229, 231, 232],

        }


class TestSeqUtilsMethods(ut.TestCase):
    """
    Test methods involving the seq_utils module.
    """
    def setUp(self):
        self.test_fastas = {
            "RRE.fasta": ("RRE", SEQS["RRE"]),
        }
        self.test_multifastas = {
            "ASCII.fasta": {
                "Phrase1": "Itwasthebestoftimes,itwastheworstoftimes.",
                "Phrase2": "Itwastheageofwisdom,itwastheageoffoolishness.",
                "Phrase3": "Itwastheepochofbelief,itwastheepochofincredulity.",
            },
        }
        self.ac_starts = [1, 100, -100]

    def test_get_kmers(self):
        seq1 = "ABCDABCD"
        k1 = 4
        true_kmers1 = {"ABCD": 2, "BCDA": 1, "CDAB": 1, "DABC": 1}
        kmers1 = dict(seq_utils.get_kmers(seq1, k1))
        self.assertEqual(kmers1, true_kmers1)

    def test_get_hamming_dist(self):
        seq11 = "ABCDEFGH"
        seq12 = "ABcDEfgH"
        true_dist1 = 3
        dist1 = seq_utils.get_hamming_dist(seq11, seq12)
        self.assertEqual(dist1, true_dist1)

    def test_read_fasta(self):
        """
        Ensure that the sequences read from a fasta file are correct.
        """
        logging.info("test_read_fasta")
        # Read single-record FASTAs and ensure the sequences are correct.
        for fasta_file, (true_name, true_seq) in self.test_fastas.items():
            name, seq = seq_utils.read_fasta(fasta_file)
            logging.debug(f"{fasta_file}\t{name}\t{true_name}\t{seq}\t{true_seq}")
            self.assertEqual(name, true_name)
            self.assertEqual(seq, true_seq)
        # Read multifastas and ensure it raises and error.
        for multifasta_file in self.test_multifastas:
            try:
                seq_utils.read_fasta(multifasta_file)
            except ValueError:
                logging.debug("successfully caught error reading multifasta with read_fasta")
            else:
                logging.debug("failed to catch error reading multifasta with read_fasta")
                self.assertTrue(False)

    def test_read_multifasta(self):
        """
        Ensure that the sequences read from a multifasta file are correct.
        """
        logging.info("test_read_multifasta")
        # Read single-record FASTAs and ensure the sequences are correct.
        for fasta_file, (true_name, true_seq) in self.test_fastas.items():
            result = seq_utils.read_multifasta(fasta_file)
            logging.debug(f"{fasta_file}\t{result}\t{true_name}\t{true_seq}")
            self.assertEqual(result, {true_name: true_seq})
        # Read multifastas.
        for multifasta_file, true_result in self.test_multifastas.items():
            result = seq_utils.read_multifasta(multifasta_file)
            logging.debug(f"{fasta_file}\t{result}\t{true_result}")
            self.assertEqual(result, true_result)

    def test_get_ac_positions(self):
        """
        Ensure that positions of A and C bases are correctly identified.
        """
        logging.info("test_get_ac_positions")
        for fasta, (name, seq) in self.test_fastas.items():
            for start in self.ac_starts:
                ac_pos = seq_utils.get_ac_positions(seq, start_pos=start)
                true_ac_pos = [pos + (start - 1) for pos in AC_POS[name]]
                logging.debug(f"{fasta}\t{name}\t{seq}\t{start}\t{ac_pos}\t{true_ac_pos}")
                self.assertEqual(ac_pos, true_ac_pos)


class TestStructUtilsMethods(ut.TestCase):
    """
    Test methods involving the structure module.
    """
    def setUp(self):
        self.rre_index = pd.Index(list(range(28, 202 + 1)), name="Position")
        self.true_rre_seq = "GCAGCAGGAAGCACUAUGGGCGCAGCGUCAAUGACGCUGACGGUACAGGCCAGACAAUUAUUGUCUGAUAUAGUGCAGCAGCAGAACAAUUUGCUGAGGGCUAUUGAGGCGCAACAGCAUCUGUUGCAACUCACAGUCUGGGGCAUCAAACAGCUCCAGGCAAGAAUCCUGGCUG"
        self.true_unpaired_1 = pd.DataFrame(np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1]]).T, index=self.rre_index, columns=["ENERGY = -144.0  RRE_28_202"])
        self.true_rre_pairs = {"ENERGY = -144.0  RRE_28_202": {(29, 202), (30, 201), (31, 200), (32, 199), (33, 197), (34, 196), (35, 195), (36, 194), (38, 103), (39, 102), (40, 101), (41, 100), (42, 99), (43, 98), (44, 97), (45, 96), (48, 69), (49, 68), (50, 66), (51, 65), (52, 64), (53, 63), (54, 62), (55, 61), (56, 60), (78, 94), (79, 93), (80, 92), (81, 91), (82, 90), (83, 89), (106, 123), (107, 122), (108, 121), (109, 120), (110, 119), (111, 118), (112, 117), (132, 160), (133, 159), (134, 158), (135, 157), (138, 154), (139, 153), (140, 152), (141, 151), (142, 150), (143, 149), (144, 148), (163, 188), (164, 187), (165, 186), (166, 185), (167, 184), (168, 183), (169, 182), (170, 181), (171, 180)}}

    def test_read_dot_file(self):
        logging.info("test_read_dot_file")
        dot_file_1 = "RRE_EM/K_1/run_1-best/210422Rou_D21-3900_RRE_28_202_InfoThresh-0.1_SigThresh-0_IncTG-NO_DMSThresh-0.5-K1_Cluster1_expUp_0_expDown_0_pk.dot"
        unpaired_1, seq = struct_utils.read_dot_file(dot_file_1, start_pos=self.rre_index[0])
        logging.debug(str(unpaired_1))
        self.assertEqual(seq, self.true_rre_seq)
        self.assertEqual(unpaired_1.shape, self.true_unpaired_1.shape)
        self.assertTrue(np.all(unpaired_1.columns == self.true_unpaired_1.columns))
        self.assertTrue(np.all(unpaired_1.index == self.true_unpaired_1.index))
        self.assertTrue(np.all(unpaired_1 == self.true_unpaired_1))

    def test_is_continuous_integers(self):
        continuous = (-3, -2, -1, 0, 1, 2)
        self.assertTrue(struct_utils.is_continuous_integers(continuous))
        not_continuous = (-3, -2, 0, 1, 2)
        self.assertFalse(struct_utils.is_continuous_integers(not_continuous))
        not_continuous = (-3, -2, 0, -1, 1, 2)
        self.assertFalse(struct_utils.is_continuous_integers(not_continuous))

    def test_read_ct_file(self):
        logging.info("test_read_ct_file")
        dot_file_1 = "RRE_EM/K_1/run_1-best/210422Rou_D21-3900_RRE_28_202_InfoThresh-0.1_SigThresh-0_IncTG-NO_DMSThresh-0.5-K1_Cluster1_expUp_0_expDown_0_pk.ct"
        pairs_1, unpaired_1, seq = struct_utils.read_ct_file(dot_file_1, start_pos=self.rre_index[0])
        logging.debug(f"unpaired_1: {unpaired_1}")
        logging.debug(f"pairs_1: {pairs_1}")
        logging.debug(f"unpaired_1.index: {unpaired_1.index}")
        logging.debug(f"true_index: {self.true_unpaired_1.index}")
        self.assertEqual(seq, self.true_rre_seq)
        self.assertEqual(pairs_1, self.true_rre_pairs)
        self.assertEqual(unpaired_1.shape, self.true_unpaired_1.shape)
        self.assertTrue(np.all(unpaired_1.columns == self.true_unpaired_1.columns))
        self.assertTrue(np.all(unpaired_1.index == self.true_unpaired_1.index))
        self.assertTrue(np.all(unpaired_1 == self.true_unpaired_1))

    def test_write_ct_file(self):
        """
        Structures
        I1: 1   5   10   15   20   25   30   35   40
        S1: ACGUCAGAAGGACGUUUGGGCCAUAAAUGGCACCCAUCUU
        D1: (((((.....)))))..((((((....)))..))).....
        """
        ct_file1 = os.path.join(data_dir, "test1.ct")
        seq1 = "ACGUCAGAAGGACGUUUGGGCCAUAAAUGGCACCCAUCUU"
        pairs1 = {"struct1": {(1 , 15), (2 , 14), (3 , 13), (4 , 12), (5 , 11),
                  (18, 35), (19, 34), (20, 33), (21, 30), (22, 29), (23, 28)}}
        if os.path.isfile(ct_file1):
            os.remove(ct_file1)
        self.assertFalse(os.path.exists(ct_file1))
        struct_utils.write_ct_file(ct_file1, seq1, pairs1, overwrite=False)
        self.assertTrue(os.path.exists(ct_file1))
        with open(ct_file1) as f:
            lines1 = f.read()
        true_lines1 = " 40 struct1\n  1 A  0  2 15  1\n  2 C  1  3 14  2\n  3 G  2  4 13  3\n  4 U  3  5 12  4\n  5 C  4  6 11  5\n  6 A  5  7  0  6\n  7 G  6  8  0  7\n  8 A  7  9  0  8\n  9 A  8 10  0  9\n 10 G  9 11  0 10\n 11 G 10 12  5 11\n 12 A 11 13  4 12\n 13 C 12 14  3 13\n 14 G 13 15  2 14\n 15 U 14 16  1 15\n 16 U 15 17  0 16\n 17 U 16 18  0 17\n 18 G 17 19 35 18\n 19 G 18 20 34 19\n 20 G 19 21 33 20\n 21 C 20 22 30 21\n 22 C 21 23 29 22\n 23 A 22 24 28 23\n 24 U 23 25  0 24\n 25 A 24 26  0 25\n 26 A 25 27  0 26\n 27 A 26 28  0 27\n 28 U 27 29 23 28\n 29 G 28 30 22 29\n 30 G 29 31 21 30\n 31 C 30 32  0 31\n 32 A 31 33  0 32\n 33 C 32 34 20 33\n 34 C 33 35 19 34\n 35 C 34 36 18 35\n 36 A 35 37  0 36\n 37 U 36 38  0 37\n 38 C 37 39  0 38\n 39 U 38 40  0 39\n 40 U 39 41  0 40"
        self.assertEqual(lines1, true_lines1)

    def test_read_combine_ct_files(self):
        ct_file1 = os.path.join(data_dir, "example1.ct")
        ct_file1_5p = os.path.join(data_dir, "example1_5p.ct")
        ct_file1_3p = os.path.join(data_dir, "example1_3p.ct")
        ct_files1 = {1: ct_file1_5p, 18: ct_file1_3p}
        titles_combine, pairs_combine, unpaired_combine, seq_combine = struct_utils.read_combine_ct_files(ct_files1)
        title_single, pairs_single, unpaired_single, seq_single = struct_utils.read_ct_file_single(ct_file1)
        self.assertTrue(pairs_single == pairs_combine)
        self.assertTrue(np.all(unpaired_single == unpaired_combine))
        self.assertTrue(seq_single == seq_combine)
        
    
    def test_get_mfmi(self):
        logging.info("test_get_mfmi")
        """
        Structure set 1
        I: 1   5   10   15   20   25   30   35   40
        1: (((((.....)))))..((((((....)))..))).....
        2: .((((.....)))).((((((((...)))...)))))...
        """
        pairs11 = {(1 , 15), (2 , 14), (3 , 13), (4 , 12), (5 , 11), (18, 35),
                   (19, 34), (20, 33), (21, 30), (22, 29), (23, 28)}
        pairs12 = {(2 , 14), (3 , 13), (4 , 12), (5 , 11), (16, 37), (17, 36),
                   (18, 35), (19, 34), (20, 33), (21, 29), (22, 28), (23, 27)}
        tp, fp, fn, tn, n = 7, 4, 5, 13, 40
        true_fmi_1 = np.sqrt((tp / (tp + fp)) * (tp / (tp + fn)))
        true_mfmi_1 = tn / n + (1 - tn / n) * true_fmi_1
        mfmi_1 = struct_utils.get_mfmi(pairs11, pairs12, first_idx=1, last_idx=40)
        self.assertTrue(np.isclose(mfmi_1, true_mfmi_1))


    def test_get_structural_elements(self):
        """
        Structures
        I: 1   5   10   15   20   25   30   35   40
        1: (((((.....)))))..((((((....)))..))).....
        2: .((((.<<<.))))...>>>....(((((...)))))...
        """
        pairs1 = {(1 , 15), (2 , 14), (3 , 13), (4 , 12), (5 , 11), (18, 35),
                  (19, 34), (20, 33), (21, 30), (22, 29), (23, 28)}
        pairs2 = {(2 , 14), (3 , 13), (4 , 12), (5 , 11), (7 , 20), (8 , 19),
                  (9 , 18), (25, 37), (26, 36), (27, 35), (28, 34), (29, 33)}
        true_elements_1 = {
                (1 , 15): [(1 , 15), (2 , 14), (3 , 13), (4 , 12), (5 , 11)],
                (18, 35): [(18, 35), (19, 34), (20, 33), (21, 30), (22, 29), (23, 28)],
        }
        true_elements_2 = {
                (2 , 20): [(2 , 14), (3 , 13), (4 , 12), (5 , 11), (7 , 20), (8 , 19), (9 , 18)],
                (25, 37): [(25, 37), (26, 36), (27, 35), (28, 34), (29, 33)],
        }
        elements_1 = struct_utils.get_structural_elements(pairs1)
        self.assertEqual(elements_1, true_elements_1)
        elements_2 = struct_utils.get_structural_elements(pairs2)
        self.assertEqual(elements_2, true_elements_2)


class TestChemProbUtilsMethods(ut.TestCase):
    """
    Test methods involving the chemical probing module.
    """
    def setUp(self):
        rre_index = {True: pd.Index(list(range(28, 202 + 1)), name="Position"), False: pd.Index([29, 30, 32, 33, 36, 37, 39, 40, 41, 43, 48, 50, 51, 53, 56, 57, 58, 61, 62, 64, 67, 68, 72, 73, 74, 77, 78, 79, 81, 82, 83, 84, 87, 92, 95, 97, 99, 103, 104, 106, 107, 109, 110, 112, 113, 114, 115, 116, 121, 124, 128, 130, 134, 137, 139, 140, 141, 142, 143, 145, 146, 148, 154, 155, 156, 157, 159, 160, 161, 162, 165, 171, 172, 174, 175, 176, 177, 178, 179, 181, 183, 184, 185, 188, 189, 190, 192, 193, 195, 196, 200], name="Position")}
        self.clusters_mu = {
            "RRE_EM/K_1/run_1-best/Clusters_Mu.txt": {True: pd.Series([0.0, 0.00441, 0.00629, 0.0, 0.0054, 0.03229, 0.0, 0.0, 0.0286, 0.10581, 0.0, 0.00363, 0.00465, 0.00333, 0.0, 0.01591, 0.0, 0.0, 0.0, 0.0, 0.01965, 0.0, 0.00389, 0.00494, 0.0, 0.00242, 0.0, 0.0, 0.00389, 0.14227, 0.13608, 0.0, 0.0, 0.00854, 0.00177, 0.0, 0.00158, 0.0, 0.0, 0.05165, 0.022, 0.0, 0.0, 0.0, 0.08379, 0.07049, 0.11488, 0.0, 0.0, 0.08316, 0.004, 0.0144, 0.0, 0.00794, 0.00232, 0.02628, 0.13607, 0.0, 0.0, 0.12125, 0.0, 0.0, 0.0, 0.0, 0.0015, 0.0, 0.0, 0.07194, 0.0, 0.0326, 0.0, 0.00767, 0.0, 0.0, 0.0, 0.00489, 0.15907, 0.0, 0.00868, 0.01212, 0.0, 0.00466, 0.01768, 0.0, 0.06149, 0.13548, 0.18097, 0.11988, 0.12872, 0.0, 0.0, 0.0, 0.0, 0.00348, 0.0, 0.0, 0.128, 0.0, 0.0, 0.0, 0.07128, 0.0, 0.14218, 0.0, 0.0, 0.0, 0.04133, 0.0, 0.0, 0.16932, 0.0, 0.00751, 0.02992, 0.0285, 0.00542, 0.01882, 0.0, 0.12263, 0.14201, 0.0, 0.03568, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00299, 0.07517, 0.15937, 0.04113, 0.0, 0.01951, 0.06447, 0.19345, 0.10304, 0.0, 0.0, 0.00243, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00419, 0.11007, 0.0, 0.1312, 0.12986, 0.1418, 0.14014, 0.21201, 0.11004, 0.0, 0.00299, 0.0, 0.002, 0.00172, 0.013, 0.0, 0.0, 0.00803, 0.12672, 0.10795, 0.0, 0.09662, 0.13115, 0.0, 0.0027, 0.00191, 0.0, 0.0, 0.0, 0.00582, 0.0, 0.0], index=rre_index[True], name="1"), False: pd.Series([0.00441, 0.00629, 0.0054, 0.03229, 0.0286, 0.10581, 0.00363, 0.00465, 0.00333, 0.01591, 0.01965, 0.00389, 0.00494, 0.00242, 0.00389, 0.14227, 0.13608, 0.00854, 0.00177, 0.00158, 0.05165, 0.022, 0.08379, 0.07049, 0.11488, 0.08316, 0.004, 0.0144, 0.00794, 0.00232, 0.02628, 0.13607, 0.12125, 0.0015, 0.07194, 0.0326, 0.00767, 0.00489, 0.15907, 0.00868, 0.01212, 0.00466, 0.01768, 0.06149, 0.13548, 0.18097, 0.11988, 0.12872, 0.00348, 0.128, 0.07128, 0.14218, 0.04133, 0.16932, 0.00751, 0.02992, 0.0285, 0.00542, 0.01882, 0.12263, 0.14201, 0.03568, 0.00299, 0.07517, 0.15937, 0.04113, 0.01951, 0.06447, 0.19345, 0.10304, 0.00243, 0.00419, 0.11007, 0.1312, 0.12986, 0.1418, 0.14014, 0.21201, 0.11004, 0.00299, 0.002, 0.00172, 0.013, 0.00803, 0.12672, 0.10795, 0.09662, 0.13115, 0.0027, 0.00191, 0.00582], index=rre_index[False], name="1")},
            "RRE_EM/K_2/run_1-best/Clusters_Mu.txt": {True: pd.DataFrame(np.array([[0.0, 0.00366, 0.00558, 0.0, 0.00576, 0.02241, 0.0, 0.0, 0.02695, 0.10693, 0.0, 0.00202, 0.00312, 0.00244, 0.0, 0.01773, 0.0, 0.0, 0.0, 0.0, 0.02196, 0.0, 0.00304, 0.00398, 0.0, 0.00294, 0.0, 0.0, 0.00349, 0.14269, 0.13403, 0.0, 0.0, 0.00724, 0.00126, 0.0, 0.00065, 0.0, 0.0, 0.0488, 0.01955, 0.0, 0.0, 0.0, 0.08809, 0.06886, 0.11212, 0.0, 0.0, 0.08049, 0.00473, 0.01343, 0.0, 0.009, 0.00417, 0.02735, 0.13711, 0.0, 0.0, 0.12557, 0.0, 0.0, 0.0, 0.0, 0.00207, 0.0, 0.0, 0.06657, 0.0, 0.02967, 0.0, 0.00783, 0.0, 0.0, 0.0, 0.00536, 0.15462, 0.0, 0.00391, 0.01028, 0.0, 0.00455, 0.00959, 0.0, 0.04872, 0.11749, 0.17164, 0.11055, 0.1279, 0.0, 0.0, 0.0, 0.0, 0.00314, 0.0, 0.0, 0.16806, 0.0, 0.0, 0.0, 0.00249, 0.0, 0.20444, 0.0, 0.0, 0.0, 0.10846, 0.0, 0.0, 0.01396, 0.0, 0.02311, 0.09221, 0.09212, 0.01758, 0.01218, 0.0, 0.0042, 0.10861, 0.0, 0.00225, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00576, 0.16728, 0.20146, 0.17039, 0.0, 0.07587, 0.09221, 0.16249, 0.1324, 0.0, 0.0, 0.00118, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00504, 0.11006, 0.0, 0.12702, 0.13031, 0.13935, 0.14858, 0.21654, 0.11625, 0.0, 0.00086, 0.0, 0.00122, 0.00128, 0.01058, 0.0, 0.0, 0.00365, 0.11219, 0.10348, 0.0, 0.07818, 0.14898, 0.0, 0.00275, 0.00167, 0.0, 0.0, 0.0, 0.00492, 0.0, 0.0], [0.0, 0.00469, 0.00655, 0.0, 0.00526, 0.03591, 0.0, 0.0, 0.02921, 0.10541, 0.0, 0.00422, 0.00521, 0.00365, 0.0, 0.01524, 0.0, 0.0, 0.0, 0.0, 0.01881, 0.0, 0.0042, 0.00529, 0.0, 0.00223, 0.0, 0.0, 0.00404, 0.14212, 0.13682, 0.0, 0.0, 0.00901, 0.00196, 0.0, 0.00192, 0.0, 0.0, 0.05269, 0.0229, 0.0, 0.0, 0.0, 0.08221, 0.07109, 0.11588, 0.0, 0.0, 0.08414, 0.00373, 0.01476, 0.0, 0.00755, 0.00165, 0.02589, 0.13569, 0.0, 0.0, 0.11967, 0.0, 0.0, 0.0, 0.0, 0.00129, 0.0, 0.0, 0.07391, 0.0, 0.03368, 0.0, 0.00762, 0.0, 0.0, 0.0, 0.00472, 0.1607, 0.0, 0.01044, 0.0128, 0.0, 0.0047, 0.02073, 0.0, 0.06636, 0.14228, 0.1845, 0.12342, 0.12903, 0.0, 0.0, 0.0, 0.0, 0.00359, 0.0, 0.0, 0.11335, 0.0, 0.0, 0.0, 0.09371, 0.0, 0.1174, 0.0, 0.0, 0.0, 0.01052, 0.0, 0.0, 0.21522, 0.0, 0.00158, 0.00436, 0.00763, 0.00117, 0.02144, 0.0, 0.16854, 0.15685, 0.0, 0.05048, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00222, 0.04611, 0.14743, 0.00164, 0.0, 0.00167, 0.05571, 0.20365, 0.09237, 0.0, 0.0, 0.00287, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00388, 0.11008, 0.0, 0.13272, 0.1297, 0.14268, 0.13706, 0.21037, 0.10778, 0.0, 0.00377, 0.0, 0.00229, 0.00188, 0.01389, 0.0, 0.0, 0.00967, 0.1322, 0.10963, 0.0, 0.10331, 0.12447, 0.0, 0.00268, 0.00199, 0.0, 0.0, 0.0, 0.00615, 0.0, 0.0]]).T, index=rre_index[True], columns=["1", "2"]), False: pd.DataFrame(np.array([[0.00366, 0.00558, 0.00576, 0.02241, 0.02695, 0.10693, 0.00202, 0.00312, 0.00244, 0.01773, 0.02196, 0.00304, 0.00398, 0.00294, 0.00349, 0.14269, 0.13403, 0.00724, 0.00126, 0.00065, 0.0488, 0.01955, 0.08809, 0.06886, 0.11212, 0.08049, 0.00473, 0.01343, 0.009, 0.00417, 0.02735, 0.13711, 0.12557, 0.00207, 0.06657, 0.02967, 0.00783, 0.00536, 0.15462, 0.00391, 0.01028, 0.00455, 0.00959, 0.04872, 0.11749, 0.17164, 0.11055, 0.1279, 0.00314, 0.16806, 0.00249, 0.20444, 0.10846, 0.01396, 0.02311, 0.09221, 0.09212, 0.01758, 0.01218, 0.0042, 0.10861, 0.00225, 0.00576, 0.16728, 0.20146, 0.17039, 0.07587, 0.09221, 0.16249, 0.1324, 0.00118, 0.00504, 0.11006, 0.12702, 0.13031, 0.13935, 0.14858, 0.21654, 0.11625, 0.00086, 0.00122, 0.00128, 0.01058, 0.00365, 0.11219, 0.10348, 0.07818, 0.14898, 0.00275, 0.00167, 0.00492], [0.00469, 0.00655, 0.00526, 0.03591, 0.02921, 0.10541, 0.00422, 0.00521, 0.00365, 0.01524, 0.01881, 0.0042, 0.00529, 0.00223, 0.00404, 0.14212, 0.13682, 0.00901, 0.00196, 0.00192, 0.05269, 0.0229, 0.08221, 0.07109, 0.11588, 0.08414, 0.00373, 0.01476, 0.00755, 0.00165, 0.02589, 0.13569, 0.11967, 0.00129, 0.07391, 0.03368, 0.00762, 0.00472, 0.1607, 0.01044, 0.0128, 0.0047, 0.02073, 0.06636, 0.14228, 0.1845, 0.12342, 0.12903, 0.00359, 0.11335, 0.09371, 0.1174, 0.01052, 0.21522, 0.00158, 0.00436, 0.00763, 0.00117, 0.02144, 0.16854, 0.15685, 0.05048, 0.00222, 0.04611, 0.14743, 0.00164, 0.00167, 0.05571, 0.20365, 0.09237, 0.00287, 0.00388, 0.11008, 0.13272, 0.1297, 0.14268, 0.13706, 0.21037, 0.10778, 0.00377, 0.00229, 0.00188, 0.01389, 0.00967, 0.1322, 0.10963, 0.10331, 0.12447, 0.00268, 0.00199, 0.00615]]).T, index=rre_index[False], columns=["1", "2"])},
        }

    def test_read_clusters_mu(self):
        """
        Ensure that read_clusters_mu reads the correct data with correct column
        labels and correctly excludes Gs and Us when indicated.
        """
        logging.info("test_read_clusters_mu")
        for mu_file, true_mus_all in self.clusters_mu.items():
            # Test several Clusters_Mu.txt files.
            for include_gu in [True, False]:
                # For each file, read with and without Gs and Us.
                true_mus = true_mus_all[include_gu]
                mus = dreem_utils.read_clusters_mu(mu_file, flatten=True,
                                                include_gu=include_gu,
                                                seq=SEQS["RRE"])
                logging.debug(f"{mu_file}\t{include_gu}\t{mus}\t{true_mus}")
                self.assertEqual(type(mus), type(true_mus))
                self.assertEqual(mus.shape, true_mus.shape)
                self.assertTrue(np.all(mus.index == true_mus.index))
                self.assertTrue(np.all(np.isclose(mus, true_mus)))
                if isinstance(true_mus, pd.DataFrame):
                    self.assertTrue(np.all(mus.columns == true_mus.columns))
                # Confirm an error is raised if the sequence is not given.
                try:
                    dreem_utils.read_clusters_mu(mu_file, flatten=True,
                                              include_gu=include_gu)
                except ValueError:
                    error = True
                else:
                    error = False
                self.assertEqual(error, not include_gu)
                # Confirm an error is raised if the sequence is too short.
                MAX_MU_POS = 202
                trims = [MAX_MU_POS - 1, MAX_MU_POS]
                for trim in trims:
                    try:
                        dreem_utils.read_clusters_mu(mu_file, flatten=True,
                                                  include_gu=include_gu,
                                                  seq=SEQS["RRE"][: trim])
                    except ValueError:
                        error = True
                    else:
                        error = False
                    self.assertEqual(error, trim < MAX_MU_POS and not include_gu)
    
    def test_dsa(self):
        """
        Test the data-structure agreement function.
        """
        # Toy data for trial set 1.
        test_paired_1 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        test_mus_1 = [.22, .08, .15, .07, .12, .04, .09, 0.10, .01, .03, .18]
        true_rbc_1 = 14 / 30
        rbc_1 = dreem_utils.get_data_structure_agreement("RBC", test_paired_1, test_mus_1,
                min_paired=5, min_unpaired=5)
        self.assertTrue(np.isclose(rbc_1, true_rbc_1))
        # Toy data with a tie (0.08) for trial set 2.
        test_paired_2 = test_paired_1
        test_mus_2 = [.22, .08, .15, .07, .12, .04, .08, 0.10, .01, .03, .18]
        true_rbc_2 = 15 / 30
        rbc_2 = dreem_utils.get_data_structure_agreement("RBC", test_paired_2, test_mus_2,
                min_paired=5, min_unpaired=5)
        self.assertTrue(np.isclose(rbc_2, true_rbc_2))
        # Test on real data for RRE.
        dot_file_3 = "RRE_EM/K_1/run_1-best/210422Rou_D21-3900_RRE_28_202_InfoThresh-0.1_SigThresh-0_IncTG-NO_DMSThresh-0.5-K1_Cluster1_expUp_0_expDown_0_pk.dot"
        mu_file_3 = "RRE_EM/K_1/run_1-best/Clusters_Mu.txt"
        start_pos = 28
        test_paired_3, seq = struct_utils.read_dot_file(dot_file_3,
                start_pos=start_pos)
        test_mus_3 = dreem_utils.read_clusters_mu(mu_file_3, flatten=True,
                include_gu=False, seq=seq, start_pos=start_pos)
        rbc_3 = dreem_utils.get_data_structure_agreement("RBC", test_paired_3.squeeze(),
                test_mus_3)

    def test_distance_distribution(self):
        vectors = np.array([[0.618999, 0.925319, 0.743951],
                            [0.426009, 0.410912, 0.175019],
                            [0.036399, 0.314585, 0.019382],
                            [0.181824, 0.420526, 0.218254],
                            [0.308501, 0.449007, 0.751962]])
        true_dists = {
                2: np.array([0.790913, 1.112393, 0.849879, 0.568636, 0.430462,
                             0.248169, 0.590019, 0.268183, 0.792958, 0.549274]),
                3: np.array([2.333768, 1.888961, 1.949568, 2.230455, 2.473987,
                             1.967789, 0.946814, 1.813439, 1.387462, 1.610415])
                }
        for degree, true_dists_deg in true_dists.items():
            dists = np.round(dreem_utils.distance_distribution(
                vectors, "euclidean", degree), 6)
            logging.debug(f"computed dists: {sorted(dists)}")
            logging.debug(f"true dists: {sorted(true_dists_deg)}")
            self.assertTrue(np.all(np.isclose(sorted(dists),
                            sorted(true_dists_deg))))

    def test_read_many_clusters_mu_files(self):
        spec_file = "/lab/solexa_rouskin/projects/mfallan/corona_fse_ASO-tiling-2_210519/Analysis_2/cluster_mus_entries.xlsx"
        dreem_utils.plot_mus(spec_file)

 


if __name__ == "__main__":
    """
    Run the unit tests from the command line.
    """
    os.chdir(data_dir)
    ut.main()

