"""Summarise the current common-diagnostic comparisons and optional reference."""
import csv
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from .decisions import bracket_content, evaluate, overlap, prefix
from .inputs import SAMPLES, VARIANTS, archived_inputs, file_hash


def _csv(path, rows, columns=()):
    keys = list(dict.fromkeys([*columns, *(k for row in rows for k in row)]))
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _read_position(path):
    with np.load(path, allow_pickle=False) as arrays:
        return arrays["position"]


def compare(sample, fits, out_rows):
    """Check actual candidate overlap and rank overlap at the DPD workload."""
    base = fits["DPD"]
    budget = int(base["upper"].sum())
    a = prefix(base["rank"], budget)
    empty = np.zeros(len(a), bool)
    for method, fit in fits.items():
        if method == "DPD":
            continue
        widths = fit["metallicity"]["dex_width"], base["metallicity"]["dex_width"]
        delta_width = None if None in widths else widths[0]-widths[1]
        b = fit["row"]["B95"]
        out_rows.append(dict(
            sample=sample, reference="DPD", method=method,
            B95_difference=None if b is None else b-base["row"]["B95"],
            U_difference=int(fit["upper"].sum())-budget,
            common_budget=budget,
            **{f"own_U_{k}": v for k, v in overlap(base["upper"], fit["upper"]).items()},
            **{f"matched_top_{k}": v for k, v in overlap(a, prefix(fit["rank"], budget)).items()},
            other_U_content_under_DPD=bracket_content(base["flags"], empty, fit["upper"]),
            DPD_U_content_under_other=bracket_content(fit["flags"], empty, base["upper"]),
            metallicity_width_difference_dex=delta_width,
        ))


def _number(x, digits=4):
    return "--" if x is None else f"{x:.{digits}f}"


def _write_tables(out, audits, cleanup, common, gamma):
    md = [
        "# APOGEE audit sensitivity",
        "",
        f"Gamma = {gamma:g}. The two samples are disjoint and no artificial contamination is introduced.",
        "Only samples for which all three new posterior fits passed appear in the tables.",
        "B95 is the 95% upper quantile of the fitted audit count; L and U bracket the selected set.",
        "These quantities concern fitted audit rules, not true contamination labels.",
        "Metallicity ranges measure sensitivity to cleanup in these samples, not population intervals.",
        "",
        "| Sample | Method | B95 / point | L | U | Median [M/H] | Central 95% range (dex) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    tex = [
        r"% Draft table; requires booktabs. Use only after checking diagnostics.csv.",
        r"\begin{table}[!htbp]", r"\centering", r"\small",
        r"\caption{Sensitivity of audit planning to the choice of $\phi$ in the two unmodified APOGEE samples. "
        r"$B_{0.95}$ is the fitted count quantile, $L$ and $U$ bracket the draw-specific audit set, "
        r"and the metallicity range measures sensitivity to cleanup within each sample.}",
        r"\label{tab:app-apogee-phi-audit}", r"\begin{tabular}{@{}rlrrrl@{}}", r"\toprule",
        r"Stars & Method & $B_{0.95}$ & $|L|$ & $|U|$ & Mean $[\mathrm{M}/\mathrm{H}]$ range \\",
        r"\midrule",
    ]
    joint = {(r["sample"], r["method"]): r for r in cleanup if r["rule"] == "joint"}
    for name, n in SAMPLES.items():
        if not common.get(name):
            md.append(f"| {name} ({n}) | awaiting three passing fits | -- | -- | -- | -- | -- |")
            continue
        size_tex = "$" + f"{n:,}".replace(",", "{,}") + "$"
        for i, r in enumerate(a for a in audits if a["sample"] == name):
            x = joint[name, r["method"]]
            low, med, high = (_number(x[k]) for k in ("dex_low", "dex_median", "dex_high"))
            band = f"[{low}, {high}]" if x["dex_low"] is not None else "--"
            b = r["B95"] if r["B95"] is not None else r["K_point"]
            md.append(f"| {name} ({n}) | {r['method']} | {b} | {r['L']} | {r['U']} | {med} | {band} |")
            if r["method"] in VARIANTS:
                band_tex = f"$[{low},\\,{high}]$" if x["dex_low"] is not None else "--"
                tex.append(" & ".join([size_tex if not i else "", r["method"], str(b),
                                       str(r["L"]), str(r["U"]), band_tex]) + r" \\")
        tex.append(r"\addlinespace")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    for fname, lines in (("README_results.md", md), ("appendix_table.tex", tex)):
        (out/fname).write_text("\n".join(lines)+"\n", encoding="utf-8")


def collect(out, manifest, samples, provenance, read_json, reference=None):
    target = out/"summary"
    target.mkdir(parents=True, exist_ok=True)
    audits, cleanup, members, changes, diag_rows, common_rows, counts = [], [], [], [], [], [], []
    common = {}
    for name, (ids, X, y, mh) in samples.items():
        complete = {}
        for variant in VARIANTS:
            folder = out/"fits"/name/variant
            fitfile = folder/"fit.json"
            if not fitfile.is_file():
                error = folder/"failure.json"
                diag_rows.append(dict(sample=name, method=variant, state="error" if error.is_file() else "pending",
                                      passed=False, error=read_json(error)["message"] if error.is_file() else None))
                continue
            fit = read_json(fitfile)
            d = fit["diagnostics"]
            diag_rows.append(dict(
                sample=name, method=variant, state="fitted", passed=fit["passed"], attempt=fit["attempt"]+1,
                eta=d["eta"], rhat_max=d["rhat_max"], ess_bulk_min=d["ess_bulk_min"],
                ess_tail_min=d["ess_tail_min"], divergences=d["divergences"], chains=d["chains"],
                warmup_per_chain=d["warmup_per_chain"], draws_per_chain=d["draws_per_chain"],
            ))
            if fit["passed"]:
                complete[variant] = _read_position(folder/f"attempt_{fit['attempt']:02d}"/"samples.npz")
        common[name] = len(complete) == len(VARIANTS)
        common_rows.append(dict(sample=name, n=len(ids), expected_fits=len(VARIANTS),
                                passing_fits=len(complete), common_pass=common[name],
                                unfiltered_mean_metallicity_dex=float(mh.mean())))
        if not common[name]:
            continue
        fits = {method: evaluate(X, y, mh, z) for method, z in complete.items()}
        if reference is not None:
            arc = archived_inputs(reference, name, X, y, provenance)
            f, b = arc["posterior"], arc["bootstrap"]
            fits["Archived DPD"] = evaluate(X, y, mh, f["position"])
            fits["K-F point"] = evaluate(X, y, mh, f["frequentist_position"], point=True)
            fits["K-F bootstrap"] = evaluate(X, y, mh, b["position"], saved_flags=b["selected"])
        compare(name, fits, changes)
        for method, fit in fits.items():
            row = {"sample": name, "n": len(ids), "method": method, "archived": method not in VARIANTS}
            audits.append({**row, **fit["row"]})
            for rule, field in (("joint", "metallicity"), ("count_only", "count_only")):
                cleanup.append({**row, "rule": rule, **fit[field]})
            ranking = np.argsort(fit["rank"])+1
            for i, star in enumerate(ids):
                members.append({**row, "apogee_id": star, "selection_frequency": fit["q"][i],
                                "rank": ranking[i], "in_L": int(fit["lower"][i]), "in_U": int(fit["upper"][i]),
                                "in_top_DPD_U_size": int(ranking[i] <= fits["DPD"]["row"]["U"])})
            for k, occurrences in enumerate(np.bincount(fit["counts"])):
                if occurrences:
                    counts.append({**row, "K": k, "draws": int(occurrences)})
    paths = {
        "audit_summary.csv": (audits, ("sample", "method", "B95", "L", "U")),
        "candidate_membership.csv": (members, ("sample", "method", "apogee_id", "rank")),
        "overlap_vs_dpd.csv": (changes, ("sample", "reference", "method", "common_budget")),
        "metallicity_ranges.csv": (cleanup, ("sample", "method", "rule", "dex_low", "dex_median", "dex_high")),
        "count_distribution.csv": (counts, ("sample", "method", "K", "draws")),
        "diagnostics.csv": (diag_rows, ("sample", "method", "state", "passed")),
        "common_samples.csv": (common_rows, ("sample", "common_pass")),
    }
    for fname, (rows, columns) in paths.items():
        _csv(target/fname, rows, columns)
    _write_tables(target, audits, cleanup, common, manifest["config"]["gamma"])
    _csv(target/"archive_reference.csv",
         [] if reference is None else [dict(
             sample=name, archive=Path(reference).name, archive_sha256=file_hash(reference),
             bootstrap_replicates=300, unsuccessful_optimizer_returns=meta["archive_bootstrap_unsuccessful_optimizer_returns"],
             max_projected_gradient=meta["archive_bootstrap_max_projected_gradient"],
         ) for name, meta in provenance["samples"].items()],
         ("sample", "archive", "archive_sha256", "bootstrap_replicates", "unsuccessful_optimizer_returns", "max_projected_gradient"))
    path = target/"apogee_audit_summary.zip"
    tmp = path.with_suffix(".tmp")
    with ZipFile(tmp, "w", ZIP_DEFLATED) as z:
        z.write(out/"manifest.json", "manifest.json")
        for f in (*paths, "archive_reference.csv", "README_results.md", "appendix_table.tex"):
            z.write(target/f, f)
    tmp.replace(path)
    return path, common
