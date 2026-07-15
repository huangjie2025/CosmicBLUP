#!/usr/bin/env python3
import argparse
import csv
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys


def run(binary, args, expected=0):
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    result = subprocess.run(
        [str(binary), *map(str, args)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    if result.returncode != expected:
        raise AssertionError(
            f"command returned {result.returncode}, expected {expected}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def require_files(prefix, suffixes):
    for suffix in suffixes:
        path = Path(f"{prefix}{suffix}")
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"missing or empty output: {path}")


def write_toy_inputs(work):
    pheno = work / "toy.pheno"
    pheno.write_text(
        "FID IID batch age Trait Trait2\n"
        "0 sample1 A 10 10.0 20.1\n"
        "0 sample2 A 11 11.2 21.0\n"
        "0 sample3 B 12 13.1 25.9\n"
        "0 sample4 B 13 14.0 28.2\n"
        "0 sample5 A 14 16.3 32.1\n"
        "0 sample6 B 15 17.1 34.0\n",
        encoding="utf-8",
    )
    pedigree = work / "toy.ped"
    pedigree.write_text(
        "id sire dam\n"
        "sample1 0 0\n"
        "sample2 0 0\n"
        "sample3 sample1 sample2\n"
        "sample4 sample1 sample2\n"
        "sample5 sample3 sample4\n"
        "sample6 sample3 sample4\n",
        encoding="utf-8",
    )
    return pheno, pedigree


def case_cli(binary, source, work):
    version = run(binary, ["--version"])
    assert version.stdout.strip() == "CosmicBLUP v1.0.0"
    help_result = run(binary, ["--help"])
    assert "PBLUP" in help_result.stdout and "Bayes" not in help_result.stdout
    topic = run(binary, ["--help=relationship"])
    assert "Relationship task help" in topic.stdout
    unknown = run(binary, ["--not-an-option"], expected=2)
    assert "Unknown option" in unknown.stderr
    missing = run(binary, ["--threads"], expected=2)
    assert "missing value" in missing.stderr.lower()


def case_relationship(binary, source, work):
    toy = source / "tests/fixtures/toy_plink/toy"
    pheno, _ = write_toy_inputs(work)
    text_prefix = work / "nested/text_g"
    run(binary, ["--make-matrix", "--relationship", "g", "--bfile", toy,
                 "--output-format", "txt", "--out", text_prefix])
    require_files(text_prefix, [".GA.txt", ".GA.id"])
    first = Path(f"{text_prefix}.GA.txt").read_text(encoding="utf-8").splitlines()[0].split()
    assert first[:2] == ["1", "1"], "text matrices must use 1-based numeric indices"

    binary_prefix = work / "binary_g"
    run(binary, ["--make-matrix", "--relationship", "g", "--bfile", toy,
                 "--output-format", "bin", "--out", binary_prefix])
    require_files(binary_prefix, [".GA.bin", ".GA.id"])
    magic = Path(f"{binary_prefix}.GA.bin").read_bytes()[:19]
    assert magic == b"COSMIC_UPPER_PACKED"

    solve_prefix = work / "external/solve"
    run(binary, ["--blup", "--model", "gblup", "--matrix", f"{text_prefix}.GA.txt",
                 "--matrix-id", f"{text_prefix}.GA.id", "--relationship", "g",
                 "--pheno", pheno, "--pheno-name", "Trait", "--skip-vce",
                 "--var-a", "0.3", "--var-e", "0.7", "--threads", "1",
                 "--out", solve_prefix])
    require_files(solve_prefix, [".beta", ".rand", ".log"])


def case_singletrait(binary, source, work):
    toy = source / "tests/fixtures/toy_plink/toy"
    pheno, pedigree = write_toy_inputs(work)
    common = ["--pheno", pheno, "--pheno-name", "Trait", "--skip-vce",
              "--var-a", "0.3", "--var-e", "0.7", "--threads", "1", "--no-se"]
    commands = {
        "pblup": ["--blup", "--model", "pblup", "--ped", pedigree],
        "gblup": ["--blup", "--model", "gblup", "--bfile", toy],
        "ssgblup": ["--blup", "--model", "ssgblup", "--ped", pedigree, "--bfile", toy],
    }
    for name, model_args in commands.items():
        prefix = work / name
        run(binary, [*model_args, *common, "--out", prefix])
        require_files(prefix, [".beta", ".rand", ".log"])


def case_repeatability(binary, source, work):
    example = source / "examples/repeatability_pblup"
    prefix = work / "repeatability"
    run(binary, ["--single-trait", "--model", "repeatability", "--ped",
                 example / "repeatability_demo.ped.txt", "--pheno",
                 example / "repeatability_demo.phe.txt", "--pheno-name", "milk",
                 "--dcovar", "season", "--qcovar", "age", "--pe", "--vce",
                 "--vce-mode", "ai", "--vce-max-iter", "10", "--threads", "1", "--out", prefix])
    require_files(prefix, [".vars", ".beta", ".rand", ".obsrand", ".log"])


def case_random_regression(binary, source, work):
    example = source / "examples/repeatability_pblup"
    prefix = work / "rrm"
    run(binary, ["--rrm", "--ped", example / "repeatability_demo.ped.txt",
                 "--pheno", example / "repeatability_demo.phe.txt", "--pheno-name", "milk",
                 "--dcovar", "season", "--qcovar", "age", "--time-col", "age",
                 "--rrm-order", "1", "--pe", "--vce", "--vce-mode", "exact",
                 "--threads", "1", "--out", prefix])
    require_files(prefix, [".rrm.meta.txt", ".rrcoef", ".rrk", ".rrcurve", ".log"])


def case_multitrait(binary, source, work):
    toy = source / "tests/fixtures/toy_plink/toy"
    pheno, _ = write_toy_inputs(work)
    prefix = work / "multitrait"
    run(binary, ["--multi-trait", "--model", "gblup", "--bfile", toy,
                 "--pheno", pheno, "--pheno-name", "Trait,Trait2", "--dcovar", "batch",
                 "--qcovar", "age", "--vce", "--vce-mode", "ai", "--vce-max-iter", "5",
                 "--threads", "1", "--out", prefix])
    require_files(prefix, [".mv.vars", ".beta", ".rand", ".log"])
    with Path(f"{prefix}.rand").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 6 and {"Trait", "Trait2"}.issubset(rows[0])


def case_auxiliary(binary, source, work):
    toy = source / "tests/fixtures/toy_plink/toy"
    qc_prefix = work / "qc"
    run(binary, ["--qc", "--bfile", toy, "--threads", "1", "--out", qc_prefix])
    assert list(work.glob("qc*")), "QC produced no outputs"

    summary = work / "summary.txt"
    summary.write_text(
        "SNP A1 A2 BETA\n"
        "snp1 A C 0.10\n"
        "snp2 A C -0.05\n"
        "snp3 A C 0.08\n"
        "snp4 A C 0.02\n",
        encoding="utf-8",
    )
    sblup_prefix = work / "sblup"
    run(binary, ["--sblup", "--bfile", toy, "--sumstat", summary, "--h2", "0.3",
                 "--window-num", "4", "--out", sblup_prefix])
    require_files(sblup_prefix, [".sblup.eff"])

    prediction_prefix = work / "prediction"
    run(binary, ["--pred", "--bfile", toy, "--score", f"{sblup_prefix}.sblup.eff",
                 "--out", prediction_prefix])
    require_files(prediction_prefix, [".profile"])


CASES = {
    "cli_contract": case_cli,
    "relationship_io": case_relationship,
    "singletrait_models": case_singletrait,
    "repeatability": case_repeatability,
    "random_regression": case_random_regression,
    "multitrait": case_multitrait,
    "auxiliary_tasks": case_auxiliary,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--bin", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    args = parser.parse_args()
    if args.work.exists():
        shutil.rmtree(args.work)
    args.work.mkdir(parents=True)
    CASES[args.case](args.bin.resolve(), args.source.resolve(), args.work.resolve())
    print(f"{args.case}: passed")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, ValueError) as error:
        print(f"FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
