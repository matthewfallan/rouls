"""
ROULS - Structure module

Purpose: I/O functions for RNA structures
Author: Matty Allan
Modified: 2021-05-23
"""


from collections import defaultdict
import logging
import os
import time

from tqdm import tqdm

import numpy as np
import pandas as pd

from matplotlib import patches
from matplotlib import pyplot as plt



UNPAIRED_CHAR = "."
NAME_MARK = ">"

def plot_arc(plot_file, seq, pairs, contcorr=False, contcorr_ax=None, start_pos=1, height_ratio=None, height_margin=0.05, title=None, dpi=300):
    """
    Plot an arc plot of an RNA structure.
    Same arguments as write_ct_file.
    """
    if height_ratio is None:
        height_ratio = 2 / (1 + 5**0.5)  # reciprocal of golden ratio
    n = len(seq)
    offset = start_pos - 1
    if contcorr:
        ax = contcorr_ax
        patch_dict = dict()
    else:
        fig, ax = plt.subplots()
    # Then plot an arc for each base pair.
    max_height = 0.0
    for base5, base3 in pairs:
        if contcorr:
            patch_dict[(base5, base3)] = dict()
        if not 0 < base5 < base3 <= n:
            raise ValueError("invalid base numbers")
        x5 = base5 + offset
        x3 = base3 + offset
        xy = ((x5 + x3) / 2, 0)
        width = x3 - x5
        height = height_ratio * width
        max_height = max(height / 2, max_height)
        theta1 = 0.0  # degrees
        theta2 = 180.0  # degrees
        if contcorr:
            arc_dict = patch_dict[(base5, base3)]
            arc_dict["xy"] = xy
            arc_dict["width"] = width
            arc_dict["height"] = height
            arc_dict["angle"] = 0.0
            arc_dict["theta1"] = theta1
            arc_dict["theta2"] = theta2
            arc_dict["linewidth"] = 0.25
        else:
            ax.add_patch(patches.Arc(xy, width, height, angle=0.0,
                    theta1=theta1, theta2=theta2, linewidth=0.25))
    if contcorr:
        max_height = max([patch["height"] for patch in list(patch_dict.values())])
        for pair in patch_dict:
            ax.add_patch(patches.Arc(patch_dict[pair]["xy"], patch_dict[pair]["width"], 0.8*(patch_dict[pair]["height"]/max_height), angle=patch_dict[pair]["angle"],
                    theta1=patch_dict[pair]["theta1"], theta2=patch_dict[pair]["theta2"], linewidth=patch_dict[pair]["linewidth"]))
        return ax
    ax.set_ylim((0.0, max_height * (height_margin + 1.0)))
    ax.set_yticks([])
    ax.set_xlim((start_pos, offset + n))
    ax.set_xlabel("Position")
    for spine in ["left", "top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.set_aspect(1.0)
    if title is not None:
        ax.set_title(str(title))
    if plot_file.endswith(".png"):
        plt.savefig(plot_file, dpi=dpi)
    else:
        plt.savefig(plot_file)
    plt.close()


def read_dot_file(dot_file, start_pos=1, title_mode="keep"):
    """
    Read a dot-bracket format RNA structure file.
    params:
    - dot_file (str): path to dot-bracket file
    - start_pos (int): number to assign first position in structure
    - title_mode (str): whether to 'keep' original titles (default) or 'number'
      from 0 to n. If 'keep', all titles must be unique.
    returns:
    - paired (pd.DataFrame[bool]): 1 if paired else 0; index is position,
      columns are headers of the structures in the file
    - seq (str): sequence of the RNA
    """
    headers = list()
    paired = dict()
    seq = None
    with open(dot_file) as f:
        header = f.readline().strip()
        while header != "":
            if title_mode == "keep":
                if header.startswith(NAME_MARK):
                    header = header[1:]
                if header in headers:
                    raise ValueError(f"Duplicate header: {header}")
            elif title_mode == "number":
                header = len(headers)
            headers.append(header)
            if seq is None:
                seq = f.readline().strip()
            dot_string = f.readline().strip()
            if len(dot_string) != len(seq):
                raise ValueError("dot_string and seq must have same length")
            paired[header] = {pos: base != UNPAIRED_CHAR
                    for pos, base in enumerate(dot_string, start=start_pos)}
            header = f.readline().strip()
    paired = pd.DataFrame.from_dict(paired, orient="columns").reindex(
            columns=headers)
    return paired, seq



def read_ct_file(ct_file, start_pos=1, title_mode="keep"):
    """
    Read a dot-bracket format RNA structure file.
    params:
    - dot_file (str): path to dot-bracket file
    - start_pos (int): number to assign first position in structure
    - title_mode (str): whether to 'keep' original titles (default) or 'number'
      from 0 to n. If 'keep', all titles must be unique.
    returns:
    - pairs (dict[set[tuple[int, int]]]): every base pair in each structure
    - paired (pd.DataFrame[bool]): 1 if paired else 0; index is position,
      columns are headers of the structures in the file
    - seq (str): sequence of the RNA
    """
    def read_header_line(line):
        """
        Read the number of bases and the title from the header.
        params:
        - line (str): line containing the header
        returns:
        - n_bases (int): number of bases in the structure
        - title (str): title of the structure
        """
        header = line.strip()
        n_bases = header.split()[0]
        # Remove the first number from the header line to get the title.
        title = header[len(n_bases):].lstrip()
        n_bases = int(n_bases)
        return n_bases, title
    def read_body_line(line, offset=0):
        """
        Read the indexes of the base and paired base.
        params:
        - line (str): line from the body of the file
        - offset (int): shift the base numbering by this number, to be used if
          the first base in the CT file covers only a segment of the RNA such
          that the first base in the CT file is numbered 1 but is not the first
          base in the full RNA sequence; in this case offset should be equal to
          actual position of first base in RNA sequence minus number of first
          base in CT file
        returns:
        - base (str): letter of the base in the sequence
        - idx_base (int): index of the base
        - idx_pair (int): index to which the base is paired, or 0 if unpaired
        - paired (bool): whether or not the base is paired
        """
        # Parse the line.
        idx_base, base, idx_5p, idx_3p, idx_pair, idx_nat = line.strip().split()
        # Check if the base is paired; unpaired bases have a pair value of 0.
        is_paired = idx_pair != "0"
        # Offset the value of the base index.
        idx_base = int(idx_base) + offset
        # Offset the value of the paired index if it is paired; 0 values stay.
        idx_pair = int(idx_pair) + offset * int(is_paired)
        return base, idx_base, idx_pair, is_paired
    # Compute the offset mapping numbering in the CT to the actual RNA.
    offset = int(start_pos - 1)
    # Initialized values of variables that are collected.
    titles = list()
    paired = dict()
    pairs = dict()
    seq = list()
    n_bases = None
    base_count = None
    first_structure_complete = False
    title = None
    with open(ct_file) as f:
        for line in f:
            if n_bases is None or base_count > n_bases:
                """
                This clause should be executed the first time the loop runs,
                and every time a new structure starts if the CT file has more
                than one structure.
                """
                # Set or reset the base count to 1.
                base_count = 1
                # Read the information from the header.
                n_bases_new, title = read_header_line(line)
                if n_bases is not None:
                    """
                    If this is not the first structure, make sure the number
                    of bases in the new structure matches the number in the
                    previous one. 
                    """
                    first_structure_complete = True
                    assert n_bases_new == n_bases
                n_bases = n_bases_new
                if title_mode == "keep":
                    # Ensure no structures have identical titles.
                    if title in titles:
                        raise ValueError(f"Repeated title: {title}")
                elif title_mode == "number":
                    # Convert title to position of structure (0-indexed).
                    title = len(titles)
                else:
                    raise ValueError(f"Invalid title_mode: '{title_mode}'")
                titles.append(title)
                # Initialize the pairs and the unpaired bases of this structure.
                pairs[title] = set()
                paired[title] = dict()
            else:
                """
                This clause executes whenever the base has not yet exceeded the
                total number of bases in the structure, so unread bases remain.
                """
                # Read the next line of the CT file.
                base, idx_base, idx_pair, is_paired = read_body_line(
                        line, offset)
                # Double check that the numbering is consistent.
                assert idx_base == base_count + offset
                if first_structure_complete:
                    # If the sequence has already been read once, make sure the
                    # new sequence matches the previous.
                    assert base == seq[base_count - 1]
                else:
                    # If the is the first time, add the base to the sequence.
                    seq.append(base)
                # Record whether the base is unpaired.
                paired[title][idx_base] = is_paired
                if is_paired:
                    # If the base is paired, add the pair to the set of pairs.
                    # All pairs are notated as (5' index, 3' index).
                    if idx_base < idx_pair:
                        # The base has a smaller index than its partner,
                        # so this pair has not yet been recorded.
                        pair = (idx_base, idx_pair)
                        # Ensure that it is not already in the set of pairs.
                        assert pair not in pairs[title]
                        # Then add it.
                        pairs[title].add(pair)
                    elif idx_pair < idx_base:
                        # The base has a larger index than its partner,
                        # so this pair has already been recorded.
                        pair = (idx_pair, idx_base)
                        # Ensure that the pair is already in the set of pairs.
                        assert pair in pairs[title]
                        # Ensure that the partner is not unpaired.
                        assert paired[title][idx_pair]
                    else:
                        # A base and its partner should not have the same index.
                        raise ValueError("base and pair index cannot be equal")
                # Increment the base count.
                base_count += 1
    if n_bases is None:
        # This means the file had no lines.
        raise ValueError(f"no lines in {ct_file}")
    # Ensure that the structure has been read completely.
    # base_count == n_bases + 1 since it was incremented at the end of the loop.
    assert base_count == n_bases + 1
    # Merge all the bases into one string.
    seq = "".join(seq)
    # Double check that the sequence is also complete.
    assert len(seq) == n_bases
    # Convert the record of unpaired bases into a DataFrame where
    # each column is a structure and each row is a position.
    paired = pd.DataFrame.from_dict(paired, orient="columns").reindex(
            columns=titles)
    # Ensure that the number of rows matches the number of bases.
    n_rows, n_cols = paired.shape
    assert n_rows == n_bases
    return pairs, paired, seq


def read_ct_file_single(ct_file, start_pos=1, title_mode="keep",
        multiple="raise"):
    pairs, paired, seq = read_ct_file(ct_file, start_pos=start_pos,
            title_mode=title_mode)
    names = list(pairs.keys())
    if multiple == "raise":
        if len(names) > 1:
            raise ValueError(f"Found {len(names)} structures in {ct_file}.")
        else:
            name = names[0]
    elif isinstance(multiple, int):
        try:
            name = names[multiple-1]
        except IndexError:
            raise ValueError(f"{ct_file} has {len(names)} structures"
                    f" but requested structure {multiple}")
    else:
        raise ValueError(f"multiple must be 'raise' or int, not {multiple}")
    return name, pairs[name], paired[name], seq


def read_dot_file_single(dot_file, start_pos=1, title_mode="keep",
        multiple="raise"):
    paired, seq = read_dot_file(dot_file, start_pos=start_pos,
            title_mode=title_mode)
    names = list(paired.columns)
    if multiple == "raise":
        if len(names) > 1:
            raise ValueError(f"Found {len(names)} structures in {dot_file}.")
        else:
            name = names[0]
    elif isinstance(multiple, int):
        try:
            name = names[multiple-1]
        except IndexError:
            raise ValueError(f"{dot_file} has {len(names)} structures"
                    f" but requested structure {multiple}")
    else:
        raise ValueError(f"multiple must be 'raise' or int, not {multiple}")
    return name, paired[name], seq


def is_continuous_integers(integers):
    if not isinstance(integers, list):
        integers = list(integers)
    if len(integers) == 0:
        raise ValueError("integers is empty")
    integers_range = list(range(integers[0], integers[-1] + 1))
    is_continuous = integers == integers_range
    return is_continuous


def read_combine_ct_files(ct_files, title_mode="keep"):
    """
    Read multiple CT files and merge the pairs into one set.
    params:
    - ct_files (dict[int, str]): dict of start_coord: ct_file_path
    returns:
    - pairs (set[tuple[int, int]]): set of all pairs (int, int)
    - seq (str): combined sequence
    - paired (pd.Series[int, bool]): whether each base is paired
    """
    titles_all = list()
    pairs_all = set()
    seq_all = list()
    paired_all = pd.Series()
    for start in sorted(ct_files.keys()):
        ct_file = ct_files[start]
        title, pairs, paired, seq = read_ct_file_single(ct_file,
                start_pos=start, title_mode=title_mode, multiple=0)
        if title_mode == "number":
            title = len(titles_all)
        elif title_mode == "keep":
            if title in titles_all:
                raise ValueError(f"repeated title: {title}")
        else:
            raise ValueError(f"Invalid title_mode: {title_mode}")
        titles_all.append(title)
        if any([pi in paired_all.index for pi in paired.index]):
            raise ValueError("repeated index")
        paired_all = pd.concat([paired_all, paired])
        pairs_all = pairs_all | pairs
        seq_all.append(seq)
    if not is_continuous_integers(paired_all.index):
        raise ValueError("index is not continuous")
    seq_all = "".join(seq_all)
    return titles_all, pairs_all, paired_all, seq_all


def read_structure_file(structure_file, **kwargs):
    """
    Read either a dot-bracket or connectivity table format RNA structure file
    and return the sequence and the unpaired bases. Automatically infer the
    file format and raise an error if it cannot be inferred.
    params:
    - structure_file (str): path to file of RNA structure
    - start_pos (int): the number given to the first base in the structure
      (determines the index numbering of return value unpaired)
    returns:
    - paired (pd.DataFrame[bool]): whether each base is paired (True) or
      unpaired (False); index is position and there is one column for each
      structure in structure_file
    - seq (str): sequence of the RNA
    """
    base, ext = os.path.splitext(structure_file)
    if ext in [".ct"]:
        pairs, paired, seq = read_ct_file(structure_file, **kwargs)
    elif ext in [".dot", ".dbn", ".bracket"]:
        paired, seq = read_dot_file(structure_file, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: '{ext}'")
    return paired, seq


def write_ct_file(ct_file, seq, pairs, start_pos=1, overwrite=False):
    """
    Write a CT format RNA structure file.
    params:
    - ct_file (str): path to output CT file
    - seq (str): RNA sequence
    - pairs (dict[str, set[tuple[int, int]]]): pairs in the structure
    - start_pos (int): the number given to the first base in the sequence;
      CT files must start at 1, so if start_pos != 1, then the quantity
      (start_pos - 1) is subtracted from all indexes of paired bases
    - overwrite (bool): if CT file already exists, overwrite?
    returns: None
    """
    if not isinstance(start_pos, int):
        raise ValueError(f"start_pos must be int")
    offset = 1 - start_pos
    if os.path.exists(ct_file) and not overwrite:
        raise ValueError(f"CT file {ct_file} already exists.")
    if not isinstance(seq, str):
        raise ValueError(f"seq must be str, not '{type(seq)}'")
    if not isinstance(pairs, dict):
        raise ValueError("pairs must be a dict")
    n_bases = len(seq)
    unpaired_idx = 0
    col_spacing = 1
    col_sep = " " * col_spacing
    col_width = len(str(n_bases + 1)) + col_spacing
    title_format_string = ("{n_bases:>" + str(col_width) + "}" + col_sep
            + "{struct_name}")
    idx_format_string = ("{idx:>" + str(col_width) + "}"
            + col_sep + "{base}"
            + "{idxUp:>" + str(col_width) + "}"
            + "{idxDn:>" + str(col_width) + "}"
            + "{paired_idx:>" + str(col_width) + "}"
            + "{idx:>" + str(col_width) + "}")
    lines = list()
    for struct_name, struct_pairs in pairs.items():
        if not isinstance(struct_pairs, (set, list)):
            raise ValueError("pairs for each structure must be set or list")
        # Validate the pairs.
        partners = dict()
        for pair in struct_pairs:
            if not isinstance(pair, tuple):
                raise ValueError("Each pair must be a tuple.")
            if len(pair) != 2:
                raise ValueError("Each pair must have 2 elements.")
            for idx_raw in pair:
                idx_adj = idx_raw + offset
                if not isinstance(idx_raw, int):
                    raise ValueError("Each pair must comprise integers.")
                if idx_raw == unpaired_idx:
                    raise ValueError(f"Pairs cannot contain {unpaired_idx}.")
                if idx_adj < 0:
                    raise ValueError("Index cannot be <0.")
                if idx_adj > n_bases:
                    raise ValueError("Index cannot be >n_bases.") 
                if idx_adj in partners:
                    raise ValueError(f"Duplicate index: {idx_raw}")
            idx1 = pair[0] + offset
            idx2 = pair[1] + offset
            if idx1 == idx2:
                raise ValueError("Elements of a pair cannot be equal.")
            partners[idx1] = idx2
            partners[idx2] = idx1
        # Write the title line.
        lines.append(f"{col_sep}{n_bases} {struct_name}")
        # Write the body lines.
        for idx, base in enumerate(seq, start=1):
            paired_idx = partners.get(idx, unpaired_idx)
            lines.append(idx_format_string.format(idx=idx, base=base,
                idxUp=idx-1, idxDn=idx+1, paired_idx=paired_idx))
    lines = "\n".join(lines)
    with open(ct_file, "w") as f:
        f.write(lines)


def write_dot_file(dot_file, seq, pairs, start_pos=1, overwrite=False):
    """
    Write a dot-bracket format RNA structure file.
    params:
    - dot_file (str): path to output CT file
    - seq (str): RNA sequence
    - pairs (dict[str, set[tuple[int, int]]]): pairs in the structure
    - start_pos (int): the number given to the first base in the sequence;
      dot files must start at 1, so if start_pos != 1, then the quantity
      (start_pos - 1) is subtracted from all indexes of paired bases
    - overwrite (bool): if dot file already exists, overwrite?
    returns: None
    """
    if not isinstance(start_pos, int):
        raise ValueError(f"start_pos must be int")
    offset = 1 - start_pos
    if os.path.exists(dot_file) and not overwrite:
        raise ValueError(f"dot file {dot_file} already exists.")
    if not isinstance(seq, str):
        raise ValueError(f"seq must be str, not '{type(seq)}'")
    if not isinstance(pairs, dict):
        raise ValueError("pairs must be a dict")
    n_bases = len(seq)
    unpaired_idx = 0
    col_spacing = 1
    col_sep = " " * col_spacing
    col_width = len(str(n_bases + 1)) + col_spacing
    title_format_string = ("{n_bases:>" + str(col_width) + "}" + col_sep
            + "{struct_name}")
    idx_format_string = ("{idx:>" + str(col_width) + "}"
            + col_sep + "{base}"
            + "{idxUp:>" + str(col_width) + "}"
            + "{idxDn:>" + str(col_width) + "}"
            + "{paired_idx:>" + str(col_width) + "}"
            + "{idx:>" + str(col_width) + "}")
    lines = list()
    for struct_name, struct_pairs in pairs.items():
        if not isinstance(struct_pairs, (set, list)):
            raise ValueError("pairs for each structure must be set or list")
        # Validate the pairs.
        partners = dict()
        for pair in struct_pairs:
            if not isinstance(pair, tuple):
                raise ValueError("Each pair must be a tuple.")
            if len(pair) != 2:
                raise ValueError("Each pair must have 2 elements.")
            for idx_raw in pair:
                idx_adj = idx_raw + offset
                if not isinstance(idx_raw, int):
                    raise ValueError("Each pair must comprise integers.")
                if idx_raw == unpaired_idx:
                    raise ValueError(f"Pairs cannot contain {unpaired_idx}.")
                if idx_adj < 0:
                    raise ValueError("Index cannot be <0.")
                if idx_adj > n_bases:
                    raise ValueError("Index cannot be >n_bases.")
                if idx_adj in partners:
                    raise ValueError(f"Duplicate index: {idx_raw}")
            idx1 = pair[0] + offset
            idx2 = pair[1] + offset
            if idx1 == idx2:
                raise ValueError("Elements of a pair cannot be equal.")
            partners[idx1] = idx2
            partners[idx2] = idx1
        # Write the title line.
        lines.append(f">{struct_name}")
        if len(lines) == 1:
            # Write the sequence.
            lines.append(seq)
        # Write the structure.
        structure = ["." for i in seq]
        for idx1, idx2 in partners.items():
            if idx1 < idx2:
                structure[idx1 - 1], structure[idx2 - 1] = ("(", ")")
            elif idx2 < idx1:
                structure[idx2 - 1], structure[idx1 - 1] = ("(", ")")
            else:
                raise ValueError("Elements of a pair cannot be equal.")
        lines.append("".join(structure))
    lines = "\n".join(lines)
    with open(dot_file, "w") as f:
        f.write(lines)


def get_structural_elements(pairs):
    """
    Given an iterable of all pairs in a structure, find every structural
    element, defined as a maximal set of contiguous bases such that every
    base in the set lies between at least two bases that are paired
    (including the outermost base pair). Informally, a structural element is
    anything that protrudes from the main horizontal line of bases in VARNA.
    params:
    - pairs (set[tuple[int, int]]): set of all base pairs in the structure
    returns:
    - elements (dict[tuple[int, int], set[tuple[int, int]]]): dict where every
      key is the bounds of the element (5', 3') and every value is the set of
      pairs in the element.
    """
    min_dist = 4
    pairs_ordered = sorted(pairs)
    bases_checked = set()
    elements = list()
    bound3p = None
    for pair_input in pairs_ordered:
        # Validate the base pair.
        if len(pair_input) != 2:
            raise ValueError("Each pair must have length 2.")
        pair5p = min(pair_input)
        pair3p = max(pair_input)
        if pair5p <= 0:
            raise ValueError("Base indexes must be positive.")
        if pair3p - pair5p < min_dist:
            raise ValueError(f"Paired bases cannot be <{min_dist} apart.")
        if pair5p in bases_checked:
            raise ValueError(f"Duplicate base: {pair5p}")
        if pair3p in bases_checked:
            raise ValueError(f"Duplicate base: {pair3p}")
        bases_checked.add(pair5p)
        bases_checked.add(pair3p)
        pair = (pair5p, pair3p)
        # Add the pair to a new structural element if its 5' base lies after
        # the 3'-most base of the current structural element.
        if bound3p is None or pair5p > bound3p:
            elements.append(list())
        # Add the pair to the current structural element.
        elements[-1].append(pair)
        # Set the 3' bound of the element to the 3'-most base in the element.
        # If there are no pseudoknots, it is the 3' base of the first pair.
        # If there are, then it is the maximum value of all 3' bases.
        if bound3p is None:
            bound3p = pair3p
        else:
            bound3p = max(bound3p, pair3p)
    # Determine the 5' and 3' bounds of each element.
    bounds = [(min([pair5p for pair5p, pair3p in element]),
               max([pair3p for pair5p, pair3p in element]))
               for element in elements]
    # Assign the bounds to each element.
    elements = dict(zip(bounds, elements))
    return elements


def dot_to_stockholm(dot_file, sto_file, extras="raise", overwrite=False):
    """
    Convert a dot-bracket file containing one entry to stockholm alignment
    format with secondary structure (to be used with Infernal cmbuild)
    """
    if os.path.isfile(sto_file) and not overwrite:
        raise ValueError(f"{sto_file} already exists")
    title_start = ">"
    with open(dot_file) as f:
        title = f.readline().strip()
        if title.startswith(title_start):
            title = title[len(title_start):]
        seq = f.readline().strip()
        struct = f.readline().strip()
        extra_lines = list(f)
    if len(extra_lines) > 0:
        if extras == "raise":
            raise ValueError(f"{len(extra_lines)} extra lines in {dot_file}")
        elif extras == "drop":
            pass
        else:
            raise ValueError(extras)
    # Write the stockholm file.
    sto_header = "# STOCKHOLM 1.0"
    sto_ssline = "#=GC SS_cons"
    sto_indent = max(len(title), len(sto_ssline)) + 1
    title_spacer = " " * (sto_indent - len(title))
    ssline_spacer = " " * (sto_indent - len(sto_ssline))
    sto_text = (f"{sto_header}\n\n{title}{title_spacer}{seq}"
            + f"\n{sto_ssline}{ssline_spacer}{struct}\n//")
    with open(sto_file, "w") as f:
        f.write(sto_text)


def stockholm_to_fasta(sto_file, fasta_file, remove_gaps=False, 
        uppercase=False, overwrite=False):
    """
    Convert a stockholm alignment file into a fasta file.
    """
    if os.path.isfile(fasta_file) and not overwrite:
        raise ValueError(f"{fasta_file} already exists")
    seqs = dict()
    comment = "#"
    end = "//"
    title = ">"
    with open(sto_file) as f:
        for line in f:
            if line.strip() not in ["", end] and not line.startswith(comment):
                data = line.strip().split()
                name, seq = data
                if remove_gaps:
                    seq = "".join([x for x in seq
                        if x.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"])
                if uppercase:
                    seq = seq.upper()
                if name in seqs:
                    seqs[name] += seq
                else:
                    seqs[name] = seq
    text = "\n".join([f">{name}\n{seq}" for name, seq in seqs.items()])
    with open(fasta_file, "w") as f:
        f.write(text)


def get_mfmi(pairs1, pairs2, first_idx, last_idx,
        dangling="raise", external="raise", validate_order=True,
        validate_bounds=True, times=None):
    """
    Compute the modified Fowlkes-Mallowes index as a measure of similarity
    between two structures.
    The Fowlkes-Mallowes index (FMI) is the geometric mean of PPV and TPR:
    FMI = sqrt(PPV * TPR) = sqrt(TP / (TP + FP) * TP / (TP + FN))
        = TP / sqrt((TP + FP) * (TP + FN))
    Here, we let TP = number of pairs in both structures
                 FP = number of pairs unique to structure 1
                 FN = number of pairs unique to structure 2
    FMI does not consider TN (number of bases that are unpaired in both
    structures), so two structures with many unpaired bases in common but no
    base pairs in common will have an FMI of 0.
    The modified FMI (mFMI) accounts for TN by adding the fraction of bases that
    are unpaired in both structures (f_unpaired) and weighting the FMI by
    (1 - f_unpaired).
    params:
    - pairs1 (list[tuple[int, int]]): all base pair indexes in structure 1
    - pairs2 (list[tuple[int, int]]): all base pair indexes in structure 2
    - first_idx (int): first index in the structure
    - last_idx (int): last index in the structure
    - dangling (str): if a pair is dangling (only one partner is in indexes),
      whether to 'raise' and error (default), 'drop' the pair, or 'keep' the
      pair (in which case it is treated like any fully internal pair)
    - external (str): if a pair is external (neither partner is in indexes),
      whether to 'raise' and error (default) or 'drop' the pair
    - validate (bool): whether to validate that the pairs are sorted and
      in bounds. If False, disregard dangling and external.
    returns:
    - mfmi (float): value of modified Fowlkes-Mallowes index
    """
    start = time.time()
    unpaired_idx = 0
    if first_idx <= 0 or last_idx <= 0:
        raise ValueError("index bounds must be positive")
    if first_idx > last_idx:
        raise ValueError("last_idx must be >= first_idx")
    n_total = last_idx - first_idx + 1
    end = time.time()
    if times is not None:
        times["checks"] += end - start
    start = time.time()
    if validate_bounds:
        def keep_pair(pair):
            if pair[0] <= 0 or pair[1] <= 0:
                raise ValueError("indexes must be positive")
            inclusion = (int(first_idx <= pair[0] <= last_idx)
                       + int(first_idx <= pair[1] <= last_idx))
            if inclusion == 2:
                # pair is internal: always keep
                keep = True
            elif inclusion == 1:
                # pair is dangling
                if dangling == "raise":
                    raise ValueError(f"Dangling pair: {pair}")
                elif dangling == "drop":
                    keep = False
                elif dangling == "keep":
                    keep = True
                else:
                    raise ValueError(f"Unexpected value for dangling: {dangling}")
            else:
                # pair is external
                if external == "raise":
                    raise ValueError(f"External pair: {pair}")
                elif external == "drop":
                    keep = False
                else:
                    raise ValueError(f"Unexpected value for dangling: {dangling}")
            return keep
    if validate_order:
        def sort_pair(pair):
            if pair[0] < pair[1]:
                return pair
            elif pair[1] < pair[0]:
                return pair[1], pair[0]
            else:
                raise ValueError("pair elements cannot be equal")
    if validate_order and validate_bounds:
        pairs1 = {sort_pair(pair) for pair in pairs1 if keep_pair(pair)}
        pairs2 = {sort_pair(pair) for pair in pairs2 if keep_pair(pair)}
    elif validate_bounds:
        pairs1 = {pair for pair in pairs1 if keep_pair(pair)}
        pairs2 = {pair for pair in pairs2 if keep_pair(pair)}
    elif validate_order:
        pairs1 = set(map(sort_pair, pairs1))
        pairs2 = set(map(sort_pair, pairs2))
    else:
        if not isinstance(pairs1, set):
            pairs1 = set(pairs1)
        if not isinstance(pairs2, set):
            pairs2 = set(pairs2)
    end = time.time()
    if times is not None:
        times["validation"] += end - start
    start = time.time()
    n_pairs_uniq1 = len(pairs1 - pairs2)
    n_pairs_uniq2 = len(pairs2 - pairs1)
    n_pairs_both = len(pairs1 & pairs2)
    if n_pairs_both > 0:
        fmi = n_pairs_both / np.sqrt((n_pairs_both + n_pairs_uniq1) *
                                     (n_pairs_both + n_pairs_uniq2))
    else:
        fmi = 0.0
    end = time.time()
    if times is not None:
        times["fmi"] += end - start
    start = time.time()
    paired = {idx for pairs in [pairs1, pairs2] for pair in pairs
            for idx in pair}
    n_unpaired_both = len({idx for idx in range(first_idx, last_idx + 1)
            if idx not in paired})
    f_unpaired = n_unpaired_both / n_total
    mfmi = f_unpaired + (1 - f_unpaired) * fmi
    end = time.time()
    if times is not None:
        times["mfmi"] += end - start
    return mfmi


def get_mfmi_windows(window_size, window_step, pairs1, pairs2,
        first_idx, last_idx, validate_order=True):
    n_total = last_idx - first_idx + 1
    times = defaultdict(float)
    if window_size > n_total:
        raise ValueError("window_size cannot exceed length of index")
    if window_step > n_total:
        raise ValueError("window_step cannot exceed length of index")
    window_starts = np.arange(first_idx, last_idx - (window_size - 1) + 1,
            window_step, dtype=np.int)
    window_ends = window_starts + (window_size - 1)
    window_frames = list(zip(window_starts, window_ends))
    mfmis = pd.Series(index=pd.MultiIndex.from_tuples(window_frames),
            dtype=np.float32)
    if validate_order:
        if any((pair[0] > pair[1] for pairs in [pairs1, pairs2] for pair in pairs)):
            raise ValueError("pair out of order")
    sorted_pairs1 = sorted(pairs1)
    first_to_sorted_idx1 = dict()
    last_to_sorted_idx1 = dict()
    for idx, pair in enumerate(sorted_pairs1):
        first_to_sorted_idx1[pair[0]] = idx
        last_to_sorted_idx1[pair[1]] = idx
    sorted_pairs2 = sorted(pairs2)
    first_to_sorted_idx2 = dict()
    last_to_sorted_idx2 = dict()
    for idx, pair in enumerate(sorted_pairs2):
        first_to_sorted_idx2[pair[0]] = idx
        last_to_sorted_idx2[pair[1]] = idx
    for win_s, win_e in tqdm(window_frames):
        idx1_min = len(sorted_pairs1)
        idx1_max = 0
        idx2_min = len(sorted_pairs2)
        idx2_max = 0
        for idx in range(win_s, win_e):
            idx1_first = first_to_sorted_idx1.get(idx, np.nan)
            if not np.isnan(idx1_first):
                if idx1_min is None or idx1_first < idx1_min:
                    idx1_min = idx1_first
                if idx1_max is None or idx1_first > idx1_max:
                    idx1_max = idx1_first
            idx1_last = last_to_sorted_idx1.get(idx, np.nan)
            if not np.isnan(idx1_last):
                if idx1_min is None or idx1_last < idx1_min:
                    idx1_min = idx1_last
                if idx1_max is None or idx1_last > idx1_max:
                    idx1_max = idx1_last
            idx2_first = first_to_sorted_idx2.get(idx, np.nan)
            if not np.isnan(idx2_first):
                if idx2_min is None or idx2_first < idx2_min:
                    idx2_min = idx2_first
                if idx2_max is None or idx2_first > idx2_max:
                    idx2_max = idx2_first
            idx2_last = last_to_sorted_idx2.get(idx, np.nan)
            if not np.isnan(idx2_last):
                if idx2_min is None or idx2_last < idx2_min:
                    idx2_min = idx2_last
                if idx2_max is None or idx2_last > idx2_max:
                    idx2_max = idx2_last
        pairs1 = sorted_pairs1[idx1_min: idx1_max + 1]
        pairs2 = sorted_pairs2[idx2_min: idx2_max + 1]
        mfmi_window = get_mfmi(pairs1, pairs2, first_idx=win_s,
                last_idx=win_e, dangling="keep", external="drop", times=times, validate_order=False)
        mfmis.loc[(win_s, win_e)] = mfmi_window
    return mfmis


def predict_structure(name, seq, mus, output_prefix, normbases,
        program="ShapeKnots", start_pos=1, winsorize=1.0, overwrite=False,
        queue=False, draw=True):
    n_bases = np.sum(np.logical_not(np.isnan(mus)))
    if n_bases == 0:
        raise ValueError("found no numerical signal in mus")
    if isinstance(normbases, float):
        if 0.0 < normbases < 1.0:
            normbases = int(round(normbases * n_bases))
        else:
            raise ValueError("If normbases is a float, it must be >0 and <1")
    if isinstance(normbases, int):
        if normbases <= 0 or normbases >= n_bases:
            raise ValueError("normbases cannot be <=0 or greater than number"
                    " of bases with signal")
    else:
        raise ValueError("normbases must be int or float")
    output_fasta = f"{output_prefix}.fasta"
    output_constraints = f"{output_prefix}.txt"
    output_ct = f"{output_prefix}.ct"
    output_ps = f"{output_prefix}.ps"
    if (any([os.path.exists(output) for output in [output_fasta, output_ct]])
            and not overwrite):
        raise ValueError(f"output files {output_prefix} exist")
    with open(output_fasta, "w") as f:
        f.write(f">{name}\n{seq}")
    # Normalize signal
    median_mu = np.median(sorted(mus)[-normbases:])
    if np.isnan(median_mu) or median_mu <= 0:
        raise ValueError(f"median mu is {median_mu}")
    norm_mus = mus / median_mu
    if isinstance(winsorize, float):
        if winsorize >= 1.0:
            norm_mus = np.minimum(norm_mus, winsorize)
        else:
            raise ValueError("winsorization value must be >=1.0")
    null_constraint = -999
    constraint_vals = list()
    for i, base in enumerate(seq, start=start_pos):
        try:
            norm_mu = norm_mus.loc[i]
        except KeyError:
            val = null_constraint
        else:
            if np.isnan(norm_mu) or norm_mu < 0.0:
                val = null_constraint
            else:
                val = norm_mu
        constraint_vals.append(val)
    with open(output_constraints, "w") as f:
        f.write("\n".join([f"{i}\t{v}"
            for i, v in enumerate(constraint_vals, start=1)]))
    cmd_fold = f"{program} {output_fasta} {output_ct} -dms {output_constraints}"
    if queue:
        cmd_fold = f"bsub -q {queue} {cmd}"
    i = os.system(cmd_fold)
    if i != 0:
        raise ValueError(f"{cmd_fold} returned exit status {i}")
    if draw:
        cmd_draw = f"draw {output_ct} {output_ps} -s {output_constraints}"
        i = os.system(cmd_draw)
        if i != 0:
            raise ValueError(f"{cmd_draw} returned exit status {i}")
    
