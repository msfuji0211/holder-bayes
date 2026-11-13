#!/usr/bin/env python3
"""Compare saved APOGEE candidate rankings and join catalogue metadata."""
import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from hashlib import sha256
import io
import json
from pathlib import Path
from zipfile import BadZipFile, ZipFile

DATA = Path(__file__).resolve().parents[1] / "data" / "apogee_audit"
MEMBERSHIP = "candidate_membership.csv"
DPD = "DPD"
COMPARISONS = ("K-F point", "K-F bootstrap")
MARKERS = (
    "gaia_dr3_source_id", "star_bad", "gaia_ruwe", "gaia_non_single_star",
    "apogee_nvisits", "apogee_vscatter_kms", "apogee_starflag", "aspcapflag",
)
OVERLAP_FIELDS = (
    "sample", "n", "comparison", "budget_rule", "matched_budget", "n_dpd",
    "n_comparison", "n_shared", "n_dpd_only", "n_comparison_only",
)
CASE_FIELDS = (
    "sample", "n", "comparison", "budget_rule", "matched_budget", "side", "apogee_id",
    "dpd_rank", "comparison_rank", "dpd_selection_frequency", "comparison_selection_frequency",
    "dpd_in_U", "comparison_in_U", *MARKERS,
)


@dataclass(frozen=True)
class Candidate:
    rank: int
    frequency: float
    in_upper: int


def read_catalogue():
    path = DATA / "catalogue_checks.csv"
    source = json.loads((DATA / "catalogue_checks_source.json").read_text(encoding="utf-8"))
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != source["csv_sha256"]:
        raise ValueError(f"Unexpected contents of {path}; use the catalogue supplied with the add-on")
    catalogue = defaultdict(dict)
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")):
        sample, star = row["sample"], row["apogee_id"]
        if star in catalogue[sample]:
            raise ValueError(f"Duplicate catalogue entry: {sample}, {star}")
        catalogue[sample][star] = {k: row[k] for k in MARKERS}
    return catalogue


def membership_rows(path):
    if path.is_dir():
        with (path / MEMBERSHIP).open(encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f)
    else:
        with ZipFile(path) as z:
            if MEMBERSHIP not in z.namelist():
                raise ValueError(f"{path} does not contain {MEMBERSHIP} at its root")
            with z.open(MEMBERSHIP) as raw, io.TextIOWrapper(raw, encoding="utf-8", newline="") as f:
                yield from csv.DictReader(f)


def read_rankings(path):
    rankings = defaultdict(lambda: defaultdict(dict))
    for row in membership_rows(path):
        sample, method, star = row["sample"], row["method"], row["apogee_id"]
        if method not in (DPD, *COMPARISONS):
            continue
        table = rankings[sample][method]
        if star in table:
            raise ValueError(f"Duplicate ranking entry: {sample}, {method}, {star}")
        c = Candidate(int(row["rank"]), float(row["selection_frequency"]), int(row["in_U"]))
        if not 0 <= c.frequency <= 1 or c.in_upper not in (0, 1):
            raise ValueError(f"Invalid selection frequency or U membership: {sample}, {method}, {star}")
        table[star] = c
    if not rankings:
        raise ValueError("No APOGEE rankings found; check that at least one sample passed all diagnostics")
    return rankings


def compare(rankings, catalogue):
    overlap_rows, case_rows = [], []
    for sample, methods in rankings.items():
        missing = [m for m in (DPD, *COMPARISONS) if m not in methods]
        if missing:
            raise ValueError(f"{sample}: missing {', '.join(missing)}. Run run_apogee_audit.py "
                             "summarize with --reference-zip and use the regenerated summary.")
        stars = catalogue.get(sample, {})
        n = len(stars)
        for method, fit in methods.items():
            if fit.keys() != stars.keys() or sorted(c.rank for c in fit.values()) != list(range(1, n + 1)):
                raise ValueError(f"Catalogue IDs or complete ranks do not match: {sample}, {method}")
        dpd = methods[DPD]
        dpd_upper = {star for star, c in dpd.items() if c.in_upper}
        for method in COMPARISONS:
            other = methods[method]
            other_upper = {star for star, c in other.items() if c.in_upper}
            budgets = (("top_10_percent", (n + 9) // 10),
                       ("matched_dpd_U", len(dpd_upper)), ("own_U", None))
            for rule, budget in budgets:
                if budget is None:
                    left, right = dpd_upper, other_upper
                else:
                    left = {star for star, c in dpd.items() if c.rank <= budget}
                    right = {star for star, c in other.items() if c.rank <= budget}
                left_only, right_only = left - right, right - left
                base = dict(sample=sample, n=n, comparison=method, budget_rule=rule, matched_budget=budget)
                overlap_rows.append(dict(
                    **base, n_dpd=len(left), n_comparison=len(right), n_shared=len(left & right),
                    n_dpd_only=len(left_only), n_comparison_only=len(right_only),
                ))
                if budget is None:
                    continue
                for side, ids, selected in (("dpd_only", left_only, dpd),
                                             ("comparison_only", right_only, other)):
                    for star in sorted(ids, key=lambda x: selected[x].rank):
                        a, b = dpd[star], other[star]
                        case_rows.append(dict(
                            **base, side=side, apogee_id=star,
                            dpd_rank=a.rank, comparison_rank=b.rank,
                            dpd_selection_frequency=a.frequency, comparison_selection_frequency=b.frequency,
                            dpd_in_U=a.in_upper, comparison_in_U=b.in_upper, **stars[star],
                        ))
    return overlap_rows, case_rows


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True,
                        help="Saved APOGEE summary ZIP or directory containing candidate_membership.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the two comparison CSVs")
    args = parser.parse_args(argv)
    try:
        overlap, cases = compare(read_rankings(args.summary.expanduser()), read_catalogue())
        out = args.output_dir.expanduser()
        out.mkdir(parents=True, exist_ok=True)
        write_csv(out / "candidate_overlap.csv", OVERLAP_FIELDS, overlap)
        write_csv(out / "discordant_candidates.csv", CASE_FIELDS, cases)
    except (OSError, ValueError, TypeError, KeyError, BadZipFile, csv.Error) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    for row in overlap:
        print("{sample}: DPD vs {comparison}, {budget_rule}: "
              "{n_shared} shared ({n_dpd} DPD, {n_comparison} comparator)".format(**row))
    print(f"Wrote {out / 'candidate_overlap.csv'} and {out / 'discordant_candidates.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
