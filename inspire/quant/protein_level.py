from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path

import pandas as pd

from inspire.constants import (
    ACCESSION_KEY,
    PEPTIDE_KEY,
)


__all__ = [
    "quantify_protein_lfq",
    "quantify_protein_ibaq",
]

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")

warnings.filterwarnings(
    "ignore",
    "Dependency 'dask' not installed",
    category=UserWarning,
    module=r"directlfq.*",
)

warnings.filterwarnings("ignore", category=UserWarning, module=r"matplotlib.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"seaborn.*|ibaqpy.*")

warnings.filterwarnings("ignore", category=DeprecationWarning) # some function calls are deprecated
warnings.filterwarnings("ignore", category=ResourceWarning) # a file not being closed explicitly after read

logging.getLogger("matplotlib.category").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s :: %(message)s",
        datefmt="%Y‑%m‑%d %H:%M:%S",
        level=logging.INFO,
    )


def _patch_single_protein_id(name):
    """
    Ibaq tokenizer expects protein ids to be separated by | and be located on the second position
    Example: sp|<id>|<gene>
    """
    return f"xx|{name}" if isinstance(name, str) and name.count("|") == 1 else name

def patch_protein_id(name):
    """
    Ibaq expects protein groups to be separated by ;  and list all protein ids in the group
    Example: <id1>;<id2>;<id3>
    """
    return ";".join(_patch_single_protein_id(tok) for tok in str(name).split()) if name else name


def extract_accessions_from_group(name):
    if not isinstance(name, str) or not name.strip():
        return name
    accs = []
    for token in name.replace(" ", ";").split(";"):
        token = token.strip()
        parts = token.split("|")
        if len(parts) >= 2:
            accs.append(parts[1])
        else:
            accs.append(token)
    return ";".join(accs)



def modify_fasta(src, dst):
    """
    Reads a FASTA at 'src', patches entries for ibaqpy,
    writes valid entries to 'dst', and drops any entry containing
    non-canonical amino acids—saving those to 'dst.excluded'.
    """
    changed = 0
    dropped = 0
    out_path = Path(dst)
    excl_path = out_path.with_suffix(out_path.suffix + ".excluded")

    with Path(src).open("r") as fin, \
         out_path.open("w") as fout, \
         excl_path.open("w") as fex:

        current_header = None
        current_seq_lines = []

        def write_current(entry_header, entry_seq, to_excluded=False):
            nonlocal changed, dropped
            # if excluded, write to excluded file
            handle = fex if to_excluded else fout

            # patch header if not excluded
            if not to_excluded:
                header_id, *rest = entry_header[1:].rstrip("\n").split(maxsplit=1)
                patched = patch_protein_id(header_id)
                if patched != header_id:
                    changed += 1
                rest_str = f" {rest[0]}" if rest else ""
                handle.write(f">{patched}{rest_str}\n")
            else:
                # keep original header
                handle.write(entry_header)

            # write sequence lines
            for seq_line in entry_seq:
                handle.write(seq_line)
            
            if to_excluded:
                dropped += 1

        for line in fin:
            if line.startswith(">"):
                # process previous entry if present
                if current_header is not None:
                    seq = "".join(l.strip() for l in current_seq_lines)
                    if set(seq).issubset(CANONICAL_AA):
                        write_current(current_header, current_seq_lines, to_excluded=False)
                    else:
                        write_current(current_header, current_seq_lines, to_excluded=True)
                # start new entry
                current_header = line
                current_seq_lines = []
            else:
                current_seq_lines.append(line)

        # process last entry
        if current_header is not None:
            seq = "".join(l.strip() for l in current_seq_lines)
            if set(seq).issubset(CANONICAL_AA):
                write_current(current_header, current_seq_lines, to_excluded=False)
            else:
                write_current(current_header, current_seq_lines, to_excluded=True)

    logger.info("Patched %d FASTA header(s)", changed)
    if dropped:
        logger.warning("Dropped %d entries containing non-canonical amino acids; see %s", dropped, excl_path)
    else:
        logger.info("No entries dropped for non-canonical amino acids")

def _parse_condition_and_rep(colname):
    m = re.match(r"(.*?)(\d+)$", colname)
    return ((m.group(1).strip() or "Condition", int(m.group(2))) if m else (colname, 1))


def prepare_ibaq_input(assignments_csv, quant_csv, output_csv, *, sample_prefix="1"):
    df_assign = pd.read_csv(assignments_csv)
    df_quant = pd.read_csv(quant_csv)

    df_assign.rename(columns={"ion": PEPTIDE_KEY}, inplace=True)
    df_quant.rename(columns={"protein": ACCESSION_KEY, "ion": PEPTIDE_KEY}, inplace=True)

    for df in (df_assign, df_quant):
        df["proteins"] = df["proteins"].apply(patch_protein_id)

    id_cols = ["proteins", "peptide"]
    value_cols = df_quant.columns.difference(id_cols)

    df_melt = (
        df_quant.melt(id_vars=id_cols, value_vars=value_cols, var_name="file_col", value_name="NormIntensity")
        .assign(file_col=lambda d: d["file_col"].str.replace("_norm$", "", regex=True))
    )
    df_melt[["Condition", "replicate"]] = df_melt["file_col"].apply(lambda s: pd.Series(_parse_condition_and_rep(s)))
    df_melt["SampleID"] = sample_prefix + df_melt["replicate"].astype(str)
    df_melt["BioReplicate"] = df_melt["replicate"]

    df_merged = df_melt.merge(df_assign, how="left", on=["peptide", "proteins"])
    df_merged["ProteinName"] = df_merged["proteins"].apply(extract_accessions_from_group)
    df_merged["PeptideCanonical"] = df_merged["peptide"]

    final_cols = [
        "ProteinName",
        "PeptideCanonical",
        "SampleID",
        "BioReplicate",
        "Condition",
        "NormIntensity",
    ]
    df_final = df_merged[final_cols].copy()

    before = len(df_final)
    df_final["NormIntensity"] = 2 ** df_final["NormIntensity"]
    df_final.dropna(subset=["NormIntensity"], inplace=True)
    dropped = before - len(df_final)
    percent_dropped = (dropped / before * 100) if before else 0
    logger.info("iBAQ input: dropped %d missing intensity row(s) (%.2f%%)", dropped, percent_dropped)

    df_final.to_csv(output_csv, index=False)
    logger.info("Saved iBAQ input to %s", output_csv)

def _lazy_import_directlfq():  
    from directlfq import lfq_manager

    return lfq_manager


def _lazy_import_ibaqpy():
    from ibaqpy.ibaq.peptides2protein import peptides_to_protein

    return peptides_to_protein


def quantify_protein_lfq(config):
    """Prepare input & run DirectLFQ (protein level)."""

    lfq_manager = _lazy_import_directlfq()

    logger.info("Running DirectLFQ quantification (protein level)")
    root = Path(config.output_folder).resolve()
    qt_dir = root / "quant" # FIXME quant_test in old config
    lfq_dir = qt_dir / "lfq"
    lfq_dir.mkdir(parents=True, exist_ok=True)

    input_csv = qt_dir / "normalised_quantification.csv"
    lfq_tsv = lfq_dir / "protein_quantification_normalized.lfq.aq_reformat.tsv"

    df = pd.read_csv(input_csv)
    df.rename(columns={"proteins": "protein", "peptide": "ion"}, inplace=True)
    df["protein"] = df["protein"].apply(patch_protein_id)
    df[df.columns.difference(["protein", "ion"])] = 2 ** df[df.columns.difference(["protein", "ion"])]
    df.to_csv(lfq_tsv, sep="\t", index=False)
    logger.info("Prepared DirectLFQ input at %s", lfq_tsv)

    lfq_manager.run_lfq(str(lfq_tsv))
    logger.info("DirectLFQ quantification completed (results dir: %s)", lfq_dir)


def quantify_protein_ibaq(config):
    """Prepare inputs & run iBAQ."""

    if not all([config.ibaq_organism, config.ibaq_ploidy]):
        raise ValueError(
            "For ibaqpy quantification, config values 'ibaqOrganism' and "
            "'ibaqPloidy' must be specified."
        )

    logger.info("Running ibaqpy quantification")
    root = Path(config.output_folder).resolve()
    qt_dir  = root / "quant"
    ibaq_dir = qt_dir / "ibaq"
    ibaq_dir.mkdir(parents=True, exist_ok=True)

    peptide_assignments = root / "finalPeptideAssignments.csv"
    quant_csv           = qt_dir / "normalised_quantification.csv"

    ibaq_input_csv = ibaq_dir / "peptides2protein_input.csv"
    fasta_mod      = ibaq_dir / "proteome_mod.fasta"
    ibaq_out       = ibaq_dir / "proteins_ibaq_output.tsv"
    qc_pdf         = ibaq_dir / "QCprofile.pdf"

    prepare_ibaq_input(peptide_assignments, quant_csv, ibaq_input_csv)
    modify_fasta(config.proteome, fasta_mod)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", ResourceWarning)

        peptides_to_protein = _lazy_import_ibaqpy()

        peptides_to_protein(
            fasta=str(fasta_mod),
            peptides=str(ibaq_input_csv),
            enzyme=config.enzyme,
            normalize=config.ibaq_normalize,
            min_aa=config.ibaq_min_aa,
            max_aa=config.ibaq_max_aa,
            tpa=config.ibaq_tpa,
            ruler=config.ibaq_ruler,
            ploidy=config.ibaq_ploidy,
            cpc=config.ibaq_cpc_cellular,
            organism=config.ibaq_organism,
            output=str(ibaq_out),
            qc_report=str(qc_pdf),
            verbose=False,
        )

    logger.info("iBAQ quantification completed (output: %s)", ibaq_out)
