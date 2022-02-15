"""
ROULS - Sequence module

Purpose: I/O functions for sequences
Author: Matty Allan
Modified: 2021-05-23
"""


from collections import Counter, defaultdict, OrderedDict
import itertools
import os
import sys

from Bio import SeqIO
import matplotlib.pyplot as plt
import pandas as pd


ALPHABETS = {
    "DNA": {"A", "C", "G", "T"},
    "RNA": {"A", "C", "G", "U"},
}

CONVERT = {
    ("DNA", "RNA"): {"A": "A", "C": "C", "G": "G", "T": "U"},
    ("RNA", "DNA"): {"A": "A", "C": "C", "G": "G", "U": "T"},
}


def get_kmers(seq, k):
    """
    Get all of the subsequences of length k (k-mers) in a sequence.
    params:
    - seq (str): sequence in which to find k-mers
    - k (int > 0): length of each subsequence
    returns:
    - kmers (dict[str, int]): dict where each key is a kmer and each value is
      the number of times it occurs in the sequence
    """
    if not isinstance(k, int):
        raise ValueError("k must be int")
    if not 1 <= k <= len(seq):
        raise ValueError("k must be >= 1 and <= length of seq")
    kmers = Counter((seq[i: i + k] for i in range(len(seq) - k + 1)))
    return kmers


def get_hamming_dist(seq1, seq2):
    """
    Compute the hamming distance between two sequences.
    params:
    - seq1 (str): first sequence
    - seq2 (str): second sequence
    returns:
    - dist (int): hamming distance
    """
    if len(seq1) != len(seq2):
        raise ValueError("seq1 and seq2 must be same length")
    dist = sum((x1 != x2 for x1, x2 in zip(seq1, seq2)))
    return dist


def standardize_seq(seq_in, alphabet, errors="raise"):
    """
    Make an arbitrary sequence conform to an alphabet.
    params:
    - seq_in (str): input sequence
    - alphabet (str): alphabet to standardize the sequence to
    - mode (str): how to handle characters that can't be translated
      - "raise": raise an error (default)
      - "omit": remove the characters from the output sequence
      - "include": keep the characters in the output sequence
    returns:
    - seq_out (str): output (standardized) sequence
    """
    assert isinstance(seq_in, str)
    seq_out = list()


def read_fasta(fasta_file, standardize=None):
    """
    Read a single sequence from a fasta file.
    params:
    - fasta_file (str): path to fasta file
    - standardize (str/None): whether to standardize to "DNA" or "RNA"
      alphabet (default: None)
    returns:
    - name (str): name of the sequence
    - seq (str): sequence
    """
    records = SeqIO.parse(fasta_file, "fasta")
    record = next(records)
    try:
        record_2 = next(records)
    except StopIteration:
        pass
    else:
        raise ValueError(f"FASTA {fasta_file} has >1 record.")
    name = record.id
    seq = str(record.seq).upper()
    return name, seq


def read_multifasta(fasta_file):
    """
    Read all sequences from a fasta file.
    params:
    - fasta_file (str): path to fasta file
    returns:
    - seqs (dict[str, str]): mapping of names to sequences
    """
    records = SeqIO.parse(fasta_file, "fasta")
    seqs = dict()
    for record in records:
        name = record.id
        seq = str(record.seq).upper()
        if name in seqs:
            raise ValueError(f"Duplicate FASTA name: {name}")
        seqs[name] = seq
    return seqs


def write_fasta(fasta_file, name, seq, overwrite=False):
    if os.path.isfile(fasta_file) and not overwrite:
        raise ValueError(f"{fasta_file} already exists")
    text = f">{name}\n{seq}"
    with open(fasta_file, "w") as f:
        f.write(text)


def write_multifasta(fasta_file, seqs, overwrite=False):
    if os.path.isfile(fasta_file) and not overwrite:
        raise ValueError(f"{fasta_file} already exists")
    text = "\n".join([f">{name}\n{seq}" for name, seq in seqs.items()])
    with open(fasta_file, "w") as f:
        f.write(text)


def split_multifasta(fasta_file, overwrite=False):
    fasta_dir = os.path.dirname(fasta_file)
    seqs = read_multifasta(fasta_file)
    for name, seq in seqs.items():
        fasta_out = os.path.join(fasta_dir, f"{name}.fasta")
        write_fasta(fasta_out, name, seq, overwrite=overwrite)


info_bases = "ACGTUacgtu"
def get_info_content(seq, fraction=False, inverse=False):
    """
    Get the number of informative bases (A, C, G, T, U) in a sequence, or if
    inverse=True, the number of uninformative bases (i.e. all others).
    params:
    - seq (str): sequence
    - fraction (bool): whether to return count (False) or fraction (True) of
      informative bases
    - inverse (bool): whether to return informative (False) or uninformative
      (True) content.
    returns:
    - info_content (int/float): number (int) or fraction (float) of informative
      or uninformative bases
    """
    n = len(seq)
    counts = Counter(seq)
    info_content = sum((counts[base] for base in info_bases))
    if inverse:
        info_content = n - info_content
    if fraction:
        info_content /= n
    return info_content


def make_unique_multifasta(fasta_in, fasta_out, overwrite=False, keep="first",
        max_ns=None):
    """
    Given a multifasta, write all unique seqeunces to a new file.
    params:
    - fasta_in (str): FASTA file from which to read sequences
    - fasta_out (str): FASTA file in which to write unique sequences
    - overwrite (bool): if fasta_out already exists, whether to overwrite
    - keep (str): when removing all but one of every unique sequence, whether to
      keep the 'first' or 'last' name of the sequence or join 'all' of them
      together separated by _and_
    - max_ns (None/int/float): maximum number of non-ACGTU characters allowed
      in the sequence. If None, no limit; if int, no more than max_ns;
      if float (>=0, <=1), limit is fraction max_ns
    """
    seqs = read_multifasta(fasta_in)
    unique_seqs = dict()
    for name, seq in seqs.items():
        try:
            unique_seqs[seq].append(name)
        except KeyError:
            unique_seqs[seq] = [name]
    if keep == "first":
        unique_seqs = {names[0]: seq for seq, names in unique_seqs.items()}
    elif keep == "last":
        unique_seqs = {names[-1]: seq for seq, names in unique_seqs.items()}
    elif keep == "all":
        unique_seqs = {"_and_".join(names): seq 
                for seq, names in unique_seqs.items()}
    else:
        raise ValueError(keep)
    if max_ns is not None:
        if isinstance(max_ns, float):
            if not 0.0 <= max_ns <= 1.0:
                raise ValueError("If max_ns is float, must be >=0 and <=1")
            fraction = True
        elif isinstance(max_ns, int):
            if max_ns < 0:
                raise ValueError("If max_ns is int, must be >=0")
            fraction = False
        else:
            raise ValueError("max_ns must be int or float if given")
        unique_seqs = {name: seq for name, seq in unique_seqs.items()
                if get_info_content(seq, fraction=fraction, inverse=True)
                <= max_ns}
    write_multifasta(fasta_out, unique_seqs, overwrite=overwrite)


def get_bases_positions(seq, bases, start_pos=1):
    """
    Get the positions in the sequence that match the given base(s).
    params:
    - seq (str): the sequence
    - bases (iterable): set of bases to look for
    - start_pos (int): the number to give the first position in the sequence
    """
    assert isinstance(seq, str)
    positions = [pos for pos, base in enumerate(seq, start=start_pos)
                 if base in bases]
    return positions


def get_ac_positions(seq, start_pos=1):
    """
    Get the positions in the sequence that are As or Cs.
    """
    return get_bases_positions(seq, "AC", start_pos=start_pos)


def sam_extract_full_reads(sam_file_in, sam_file_out, min_start, max_start,
        min_end, max_end, overwrite=False):
    """
    Extract all read pairs that begin within the range (min_start, max_start)
    and end within the range (min_end, max_end), inclusive of ends.
    params:
    - sam_file_in (str): path to input SAM file
    - sam_file_out (str): path to output SAM file; if it already exists,
      response depends on overwrite parameter
    - min_start (int): 5'-most position at which a read pair's 5' end can be
      located for it to be included in output
    - max_start (int): 3'-most position at which a read pair's 5' end can be
      located for it to be included in output
    - min_end (int): 5'-most position at which a read pair's 3' end can be
      located for it to be included in output
    - max_end (int): 3'-most position at which a read pair's 3' end can be
      located for it to be included in output
    - overwrite (bool): if sam_file_out already exists, whether to overwrite or
      raise an error (default)
    returns:
    - freq_start (dict[float]): frequency of each start position
    - freq_end (dict[float]): frequency of each end position
    """
    header_char = "@"
    if os.path.exists(sam_file_out):
        if not overwrite:
            raise ValueError(f"file exists: {sam_file_out}")
    with open(sam_file_in) as fi, open(sam_file_out, "w") as fo:
        freq = defaultdict(lambda: defaultdict(int))
        def process_line(line):
            """
            Process a single SAM file line.
            params:
            - line (str): line from SAM file
            returns:

            """
            info = line.split("\t")
            qname = info[0]  # query name
            pos = int(info[3])  # 5'-most mapping position
            pnext = int(info[7])  # 5'-most mapping position of mate
            tlen = int(info[8])  # template length
            return qname, pos, pnext, tlen

        def process_lines(line1, line2):
            """
            Process a line and its mate line, determine if they should be
            written to the output file; if so, write them.
            params:
            - line1 (str): line representing read 1
            - line2 (str): line representing read 2
            returns: None
            """
            # Extract info from the lines.
            qname1, pos1, pnext1, tlen1 = process_line(line1)
            qname2, pos2, pnext2, tlen2 = process_line(line2)
            # Confirm the reads are paired.
            assert qname1 == qname2
            tlen = abs(tlen1)
            assert tlen == abs(tlen2)
            assert pos1 == pnext2
            assert pos2 == pnext1
            # Compute the 5' and 3' ends of the read pair.
            pos5p = min(pos1, pos2)
            pos3p = pos5p + abs(tlen1) - 1
            freq[pos5p][pos3p] += 1
            # If the ends are in the range, write them to the output file.
            if min_start <= pos5p <= max_start and min_end <= pos3p <= max_end:
                print(pos5p, pos3p)
                fo.write(line1 + line2)

        # Write all lines in the header into the output file.
        line1 = next(fi)
        while line1.startswith(header_char):
            fo.write(line1)
            line1 = next(fi)
        line2 = next(fi)
        process_lines(line1, line2)
        for line1 in fi:
            line2 = next(fi)
            process_lines(line1, line2)
    freq = pd.DataFrame.from_dict(freq,
            orient="index").fillna(0).sort_index(0).sort_index(1)
    return freq

