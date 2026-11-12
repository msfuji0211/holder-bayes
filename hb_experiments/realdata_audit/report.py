"""Keep every planned job visible; compute comparisons on common passing jobs."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from . import design as d
from .decision import BOOT, KF

BASE = ["id", "dataset", "replicate", "shift", "gamma", "n_train", "n_test", "p",
        "injected", "injected_rate", "paired_valid", "method", "rule"]
GROUP = {
    "capacity": (["rule"], ["loss", "budget", "budget_rate", "shortage", "excess", "covers"]),
    "triage": (["rule", "false_remove_cost", "miss_cost", "review_cost", "primary_cost"],
               ["loss", "review_rate", "miss_rate", "false_removal_rate", "rmse", "mae"]),
    "removal": (["rule"], ["removed", "f1", "recall", "precision", "noninjected_removal_rate", "rmse", "mae"]),
    "counts": ([], ["epsilon_mean", "count_mean", "count_sd", "count_q025", "count_q975", "count_width"]),
}


def _frame(rows, defaults):
    columns = list(dict.fromkeys([*defaults, *(k for row in rows for k in row)]))
    return pd.DataFrame(rows, columns=columns)


def _write(frame, path):
    with d.atomic_file(path, "w") as out:
        frame.to_csv(out, index=False)


def grouped(frame, keys, metrics):
    rows = []
    if not frame.empty:
        for vals, sub in frame.groupby(keys, sort=False, dropna=False):
            row = dict(zip(keys, vals if isinstance(vals, tuple) else (vals,)), pair_n=len(sub))
            for metric in metrics:
                values = pd.to_numeric(sub[metric], errors="coerce").dropna() if metric in sub else pd.Series(dtype=float)
                row.update({f"{metric}_mean": values.mean(), f"{metric}_sd": values.std(),
                            f"{metric}_n": len(values)})
            rows.append(row)
    return _frame(rows, [*keys, "pair_n", *(f"{m}_{s}" for m in metrics for s in ("mean", "sd", "n"))])


def paired_rows(capacity, triage):
    out = []

    def add(kind, frame, keys, a, b):
        def select(method_rule):
            method, rule = method_rule
            block = frame.loc[frame.method.eq(method)]
            if rule is not None:
                block = block.loc[block.rule.eq(rule)]
            return block[["id", "replicate", "dataset", "shift", *keys, "loss"]]
        a_frame, b_frame = select(a), select(b)
        joined = a_frame.merge(b_frame, on=["id", "replicate", "dataset", "shift", *keys],
                              suffixes=("_a", "_b"), validate="one_to_one")
        for rec in joined.to_dict("records"):
            out.append(dict(kind=kind, method=a[0], method_rule=a[1], reference=b[0], reference_rule=b[1],
                            **rec, delta_loss=rec["loss_a"]-rec["loss_b"]))

    q = "cost_quantile"
    if not capacity.empty:
        for variant in d.VARIANTS:
            for ref in [(variant, "central"), (KF, "point"), (BOOT, q)] + ([("DPD", q)] if variant != "DPD" else []):
                add("capacity", capacity, [], (variant, q), ref)
    if not triage.empty:
        keys = ["rule", "false_remove_cost", "miss_cost", "review_cost", "primary_cost"]
        for variant in d.VARIANTS:
            ref = [f"{variant} central", f"{variant} count only", f"{KF} point", BOOT]
            if variant != "DPD":
                ref.append("DPD FoD")
            for other in ref:
                add("triage", triage, keys, (f"{variant} FoD", None), (other, None))
    return _frame(out, ["kind", "method", "method_rule", "reference", "reference_rule", "id",
                        "replicate", "dataset", "shift", "rule", "false_remove_cost", "miss_cost",
                        "review_cost", "primary_cost", "loss_a", "loss_b", "delta_loss"])


def _number(x):
    return "--" if x is None or pd.isna(x) else f"{x:.5f}"


def paper_tables(target, c, availability, cap, tri):
    columns = (KF, BOOT, "DPD", "ExpSat", "RatSat")
    costs = c["costs"]
    quantile = 1-costs["review"]/costs["shortage"]
    cap = cap.loc[cap.rule.isin(["point", "cost_quantile"])]
    tri = tri.loc[tri.rule.eq("risk_proxy") & tri.primary_cost.eq(True)]
    cfg = f"false removal={costs['false_remove']:g}, miss={costs['miss']:g}, review={costs['review']:g}"
    lines = [
        "% Generated from the current common-passing jobs. Requires booktabs.",
        "% Draft table: interpret with completion.csv; no formal inference across overlapping splits.",
    ]
    configs = [
        (cap, "Capacity", f"Review-capacity loss (reserve cost {costs['review']:g}, shortage cost {costs['shortage']:g}, quantile {quantile:g})",
         list(columns)),
        (tri, "Triage", f"Retain/remove/review loss ({cfg})",
         [f"{KF} point", BOOT, "DPD FoD", "ExpSat FoD", "RatSat FoD"]),
    ]
    md = [
        "# Additional regression benchmark: audit decisions",
        "",
        f"Configuration: **{c['purpose']}**. Losses are per training row; lower is better.",
        "A pair is retained only if all three posteriors, the K-F fit, and all bootstrap replicates pass.",
        "A dash means that no common-passing pair was available.",
        "Shift is measured in pre-injection residual SD units. Shift zero denotes no artificial injection.",
        "",
    ]
    for frame, tag, caption, names in configs:
        lines += [
            r"\begin{table}[!htbp]", r"\centering", r"\footnotesize",
            rf"\caption{{{caption}. Mean loss per training observation. Pairs counts the common diagnostic passes.}}",
            rf"\label{{tab:realdata-audit-{tag.lower()}}}", r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
            r"Dataset & Shift & Pairs & K-F & Bootstrap & DPD & ExpSat & RatSat \\", r"\midrule",
        ]
        md += [f"## {caption}", "",
               "| Dataset | Shift | Passing / planned | K-F | Bootstrap | DPD | ExpSat | RatSat |",
               "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in availability.to_dict("records"):
            data = r["dataset"].replace("_", "-").replace("pumadyn32fh", "pumadyn-32fh")
            block = frame.loc[frame.dataset.eq(r["dataset"]) & frame["shift"].eq(r["shift"])].set_index("method")
            vals = [_number(block.loc[k, "loss_mean"]) if k in block.index else "--" for k in names]
            lines.append(" & ".join([data, f"{r['shift']:g}", str(r["passed"]), *vals]) + r" \\")
            md.append("| " + " | ".join([data, f"{r['shift']:g}", f"{r['passed']}/{r['planned']}", *vals]) + " |")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        md += ["", "K-F uses its point estimate. Bootstrap and H-B use their fitted count distribution for capacity",
               "and their selection frequencies for triage.", ""]
    for name, data in (("appendix_tables.tex", lines), ("README_results.md", md)):
        with d.atomic_file(target/name, "w") as f:
            f.write("\n".join(data) + "\n")


def collect(out):
    out = Path(out)
    manifest = d.read_json(out/"manifest.json")
    c = manifest["config"]
    target = out/"summary"
    with d.lock(out/".summary.lock"):
        completion, tables, inputs = [], {name: [] for name in GROUP}, []
        for name in c["datasets"]:
            receipt = out/"datasets"/f"{name}.json"
            if receipt.exists():
                info = d.read_json(receipt)
                inputs.append(dict(dataset=name, X_sha256=info["hashes"][0], y_sha256=info["hashes"][1],
                                   n=info["shape"][0], p=info["shape"][1]))
        for job in manifest["jobs"]:
            folder = out/"jobs"/job["id"]
            file = folder/"result.json"
            if not file.exists():
                error = folder/"failure.json"
                completion.append(dict(**job, completed=False, paired_valid=False,
                                       state="error" if error.exists() else "pending",
                                       error=d.read_json(error)["message"] if error.exists() else None))
                continue
            value = d.read_json(file)
            if value["fingerprint"] != manifest["fingerprint"] or value["job"] != job:
                raise ValueError(f"Unexpected job identity in {file}")
            status = value["status"]
            completion.append(dict(**job, completed=True, state="complete", **status))
            for name in tables:
                for metric in value[name]:
                    tables[name].append(dict(**job, **status, **metric))
        comp = _frame(completion, [*BASE[:5], "state", "completed", "paired_valid"])
        avail = comp.groupby(["dataset", "shift"], sort=False).agg(
            planned=("id", "size"), completed=("completed", "sum"), passed=("paired_valid", "sum")).reset_index()
        shift_pairs = []
        for name, frame in comp.groupby("dataset", sort=False):
            contaminated = frame.loc[frame["shift"] > 0]
            flags = contaminated.groupby("replicate")["paired_valid"].all()
            shift_pairs.append(dict(dataset=name, common_injected_splits=int(flags.sum()), planned_splits=len(flags)))
        _write(comp, target/"completion.csv")
        _write(avail, target/"availability.csv")
        _write(_frame(shift_pairs, ["dataset", "common_injected_splits", "planned_splits"]), target/"common_shifts.csv")
        _write(_frame(inputs, ["dataset", "X_sha256", "y_sha256", "n", "p"]), target/"input_datasets.csv")
        files = ["completion.csv", "availability.csv", "common_shifts.csv", "input_datasets.csv"]
        passing, summaries = {}, {}
        for name, rows in tables.items():
            keys, metrics = GROUP[name]
            frame = _frame(rows, [*BASE, *keys, *metrics])
            passing[name] = frame.loc[frame["paired_valid"].eq(True)]
            summaries[name] = grouped(passing[name], ["dataset", "shift", "method", *keys], metrics)
            for label, values in (("all", frame), ("summary", summaries[name])):
                file = f"{name}_{label}.csv"
                _write(values, target/file)
                files.append(file)
        pairs = paired_rows(passing["capacity"], passing["triage"])
        paired = grouped(pairs, ["kind", "dataset", "shift", "method", "method_rule", "reference", "reference_rule",
                                 "rule", "false_remove_cost", "miss_cost", "review_cost", "primary_cost"], ["delta_loss"])
        _write(pairs, target/"paired_differences.csv")
        _write(paired, target/"paired_summary.csv")
        paper_tables(target, c, avail, summaries["capacity"], summaries["triage"])
        files += ["paired_differences.csv", "paired_summary.csv", "appendix_tables.tex", "README_results.md"]
        bundle = target/"realdata_audit_summary.zip"
        with d.atomic_file(bundle, "wb") as f:
            with ZipFile(f, "w", ZIP_DEFLATED) as z:
                z.write(out/"manifest.json", "manifest.json")
                for name in files:
                    z.write(target/name, name)
    return bundle
