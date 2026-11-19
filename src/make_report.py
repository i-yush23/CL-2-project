"""
Generate project report and summary from results CSV files.
Renders notes_template.md to NOTES.md, writes results/summary.md,
results/derived_stats.csv, and results/before_after.csv.
"""

import glob
import json
import math
import os
import re
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.normalise import (
    HIN_SHARE_MIN,
    ENG_SHARE_MAX,
    SIMILARITY_MIN,
    MIN_FREQ_RATIO,
)


def md_table(df: pd.DataFrame) -> str:
    """Render a pandas DataFrame as a GitHub-style markdown table."""
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for col in headers:
            v = row[col]
            if pd.isna(v):
                cells.append("")
            elif isinstance(v, (int, pd.Int64Dtype)):
                cells.append(str(int(v)))
            elif isinstance(v, float):
                int_cols = ("count", "rule_tokens", "gold_O", "gold_Hin", "gold_Eng", "support", "value", "v1", "now")
                if v.is_integer() and col in int_cols:
                    cells.append(str(int(v)))
                else:
                    cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def compute_before_after(v1_dir: str, now_dir: str) -> pd.DataFrame:
    """Compare results_v1 and current results for before_after.csv."""
    rows = []
    p_v1, p_now = f"{v1_dir}/variant_dict.json", f"{now_dir}/variant_dict.json"
    if os.path.exists(p_v1) and os.path.exists(p_now):
        with open(p_v1, "r", encoding="utf-8") as f1, open(p_now, "r", encoding="utf-8") as f2:
            rows.append({"metric": "variant_entries", "v1": len(json.load(f1)), "now": len(json.load(f2))})

    p_v1, p_now = f"{v1_dir}/lid_comparison.csv", f"{now_dir}/lid_comparison.csv"
    if os.path.exists(p_v1) and os.path.exists(p_now):
        df_v1 = pd.read_csv(p_v1).set_index("condition")
        df_now = pd.read_csv(p_now).set_index("condition")
        for cond in df_now.index:
            if cond in df_v1.index:
                rows.append({"metric": f"lid_accuracy_{cond}", "v1": df_v1.loc[cond, "accuracy"], "now": df_now.loc[cond, "accuracy"]})
                rows.append({"metric": f"lid_macro_f1_{cond}", "v1": df_v1.loc[cond, "macro_f1"], "now": df_now.loc[cond, "macro_f1"]})

    p_v1, p_now = f"{v1_dir}/baseline_ablation.csv", f"{now_dir}/baseline_ablation.csv"
    if os.path.exists(p_v1) and os.path.exists(p_now):
        df_v1 = pd.read_csv(p_v1).set_index("config")
        df_now = pd.read_csv(p_now).set_index("config")
        for cfg in ["R0", "R1", "R2", "R3", "R4", "R1_raw"]:
            if cfg in df_v1.index and cfg in df_now.index and "dev_macro_f1" in df_v1.columns:
                rows.append({"metric": f"dev_macro_f1_{cfg}", "v1": df_v1.loc[cfg, "dev_macro_f1"], "now": df_now.loc[cfg, "dev_macro_f1"]})
    return pd.DataFrame(rows)


def extract_cl1_files_used(cl1_path: str) -> str:
    """Extract and sort deduplicated CL-1 modules and data JSON files."""
    with open(cl1_path, "r", encoding="utf-8") as f:
        content = f.read()
    mods = re.findall(r"(?:from|import)\s+(hinglish[\w\.]*)", content)
    files = [f"data/{m}" for m in re.findall(r"[\"']([a-zA-Z0-9_\-]+\.json)[\"']", content)]
    return ", ".join(sorted(set(mods + files)))


def compute_mcnemar(b: int, c: int) -> float:
    """Exact two-tailed binomial test for McNemar table."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, p)


def extract_normalisation_stats(stats_path: str) -> dict:
    """Read normalisation_stats.csv into dictionary."""
    df = pd.read_csv(stats_path).set_index("metric")
    get_val = lambda m: int(df.loc[m, "value"])
    res = {
        "HIN_SHARE_MIN": HIN_SHARE_MIN,
        "ENG_SHARE_MAX": ENG_SHARE_MAX,
        "SIMILARITY_MIN": SIMILARITY_MIN,
        "MIN_FREQ_RATIO": MIN_FREQ_RATIO,
        "entries": get_val("entries"),
        "entries_learned": get_val("entries_learned"),
        "entries_cl1_variants": get_val("entries_cl1_variants"),
        "entries_cl1_slang": get_val("entries_cl1_slang"),
        "distinct_canonicals": get_val("distinct_canonicals"),
        "dev_tokens_changed": get_val("dev_tokens_changed"),
        "dev_tokens_total": get_val("dev_tokens_total"),
    }
    pct = 100.0 * res["dev_tokens_changed"] / res["dev_tokens_total"]
    res["dev_changed_pct"] = f"{pct:.1f}"
    return res


def extract_lid_stats(res_dir: str) -> tuple[dict, dict]:
    """Compute LID accuracy metrics and O-rule numbers."""
    raw = pd.read_csv(f"{res_dir}/lid_raw.csv")
    norm = pd.read_csv(f"{res_dir}/lid_normalised.csv")
    cleaned = pd.read_csv(f"{res_dir}/lid_cleaned.csv")
    rule = pd.read_csv(f"{res_dir}/lid_rule_stats.csv")
    cv = pd.read_csv(f"{res_dir}/lid_cv_accuracy.csv").set_index("split")

    o_row = raw[raw["class"] == "O"].iloc[0]
    calc_he = lambda df: (
        (df[df["class"].isin(["Hin", "Eng"])]["recall"] * df[df["class"].isin(["Hin", "Eng"])]["support"]).sum()
        / df[df["class"].isin(["Hin", "Eng"])]["support"].sum()
    )
    raw_rules = int(rule[rule["condition"] == "raw"]["rule_tokens"].iloc[0])
    he_raw, he_norm, he_clean = calc_he(raw), calc_he(norm), calc_he(cleaned)

    stats = {
        "o_share_pct": f"{100.0 * o_row['support'] / raw['support'].sum():.1f}",
        "o_prec": f"{o_row['precision']:.4f}",
        "o_rec": f"{o_row['recall']:.4f}",
        "hin_eng_acc_raw": f"{he_raw:.4f}",
        "hin_eng_acc_norm": f"{he_norm:.4f}",
        "rule_tokens": raw_rules,
        "gap_tokens": int(o_row["support"] - raw_rules),
        "oof_acc": f"{cv.loc['train_oof', 'token_accuracy']:.4f}",
        "dev_lid_acc": f"{cv.loc['dev', 'token_accuracy']:.4f}",
    }
    return stats, {"raw": he_raw, "cleaned": he_clean, "normalised": he_norm}


def extract_error_summary_stats(error_path: str) -> dict:
    """Extract McNemar and change-type stats from lid_error_summary.csv."""
    df = pd.read_csv(error_path)

    def get_cnt(p, k, g, ct):
        sub = df[(df["pair"] == p) & (df["kind"] == k) & (df["gold"] == g) & (df["change_type"] == ct)]
        return int(sub["count"].iloc[0]) if len(sub) else 0

    stats = {}
    for pfx, pair in [("cn", "clean_vs_norm"), ("rn", "raw_vs_norm")]:
        h, hr = get_cnt(pair, "helped", "all", "all"), get_cnt(pair, "hurt", "all", "all")
        stats[f"{pfx}_helped"], stats[f"{pfx}_hurt"] = h, hr
        stats[f"{pfx}_net"], stats[f"{pfx}_p"] = h - hr, f"{compute_mcnemar(h, hr):.3f}"
        for ct in ["none", "case_only", "cleaning", "dictionary"]:
            ch, chr_ = get_cnt(pair, "helped", "all", ct), get_cnt(pair, "hurt", "all", ct)
            if ct == "none":
                stats[f"{pfx}_none_helped"], stats[f"{pfx}_none_hurt"] = ch, chr_
                stats[f"{pfx}_none_net"] = ch - chr_
                stats[f"{pfx}_none_p"] = f"{compute_mcnemar(ch, chr_):.3f}"
            elif ct == "dictionary":
                stats[f"{pfx}_dict_helped"], stats[f"{pfx}_dict_hurt"] = ch, chr_
            elif ct == "cleaning":
                stats[f"{pfx}_clean_net"] = ch - chr_
            elif ct == "case_only":
                stats[f"{pfx}_case_net"] = ch - chr_
        stats[f"{pfx}_hin_net"] = get_cnt(pair, "helped", "Hin", "all") - get_cnt(pair, "hurt", "Hin", "all")
        stats[f"{pfx}_eng_net"] = get_cnt(pair, "helped", "Eng", "all") - get_cnt(pair, "hurt", "Eng", "all")
    return stats


def extract_baseline_stats(res_dir: str) -> tuple[dict, str]:
    """Extract baseline ablation stats and find best configuration."""
    df_abl = pd.read_csv(f"{res_dir}/baseline_ablation.csv").set_index("config")
    df_del = pd.read_csv(f"{res_dir}/baseline_deltas.csv")

    eval_cfgs = ["R0", "R1", "R2", "R3", "R4", "R1_raw"]
    sub_abl = df_abl.loc[eval_cfgs]
    best_cfg = str(sub_abl["dev_macro_f1"].idxmax())
    best_row = sub_abl.loc[best_cfg]

    sig = df_del[(df_del["ci_low"] > 0) | (df_del["ci_high"] < 0)]["pair"].tolist()
    null = df_del[~((df_del["ci_low"] > 0) | (df_del["ci_high"] < 0))]["pair"].tolist()

    pc_files = glob.glob(f"{res_dir}/baseline_{best_cfg}_per_class.csv")
    df_pc = pd.read_csv(pc_files[0])
    weak_row = df_pc.sort_values("f1_score").iloc[0]

    r4 = df_abl.loc["R4"]
    stats = {
        "r4_f1": f"{r4['dev_macro_f1']:.4f}",
        "r4g_f1": f"{df_abl.loc['R4_gold', 'dev_macro_f1']:.4f}",
        "r4i_f1": f"{df_abl.loc['R4_insample', 'dev_macro_f1']:.4f}",
        "r4_ci_width": f"{(r4['macro_f1_ci_high'] - r4['macro_f1_ci_low']):.4f}",
        "significant_steps": ", ".join(sig) if sig else "no step",
        "null_steps": ", ".join(null),
        "best_config": best_cfg,
        "best_f1": f"{best_row['dev_macro_f1']:.4f}",
        "best_acc": f"{best_row['dev_accuracy']:.4f}",
        "weak_class": str(weak_row["class"]),
        "weak_f1": f"{weak_row['f1_score']:.4f}",
    }
    return stats, best_cfg


def check_template_digits(template_text: str) -> None:
    """Assert no hardcoded numbers remain outside permitted patterns."""
    clean = re.sub(r"\{\{.*?\}\}", "", template_text)
    clean = re.sub(r"F[1-5]|R[0-4](_\w+)?|CL-[12]|cl1|5-fold|95%|UTF-8|SentiMix", "", clean, flags=re.IGNORECASE)
    digits = re.findall(r"\d", clean)
    assert not digits, f"Digit check failed; found digits: {digits}"


def render_report(template_text: str, stats: dict, tables: dict) -> str:
    """Render template placeholders and tables into markdown text."""
    rendered = template_text
    for t_name, t_md in tables.items():
        rendered = rendered.replace(f"{{{{table:{t_name}}}}}", t_md)

    matches = re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}", rendered)
    for k in matches:
        if k not in stats:
            raise KeyError(f"Unknown placeholder: {k}")
        rendered = rendered.replace(f"{{{{{k}}}}}", str(stats[k]))

    assert "{{" not in rendered, "Unrendered placeholder remains in output"
    return rendered


def load_tables(res_dir: str, best_cfg: str, he_map: dict) -> dict:
    """Load and format all markdown tables."""
    tables = {}
    t_files = {
        "data_summary": "data_summary.csv",
        "normalisation_stats": "normalisation_stats.csv",
        "lid_comparison": "lid_comparison.csv",
        "lid_rule_stats": "lid_rule_stats.csv",
        "lid_error_summary": "lid_error_summary.csv",
        "baseline_ablation": "baseline_ablation.csv",
        "baseline_deltas": "baseline_deltas.csv",
        "best_per_class": f"baseline_{best_cfg}_per_class.csv",
        "before_after": "before_after.csv",
    }
    for name, fname in t_files.items():
        df = pd.read_csv(f"{res_dir}/{fname}")
        if name == "lid_comparison":
            df["hin_eng_acc"] = df["condition"].map(he_map)
        tables[name] = md_table(df)
    return tables


def main() -> None:
    """Execute reporting pipeline."""
    with open("notes_template.md", "r", encoding="utf-8") as f:
        template = f.read()
    check_template_digits(template)

    df_ba = compute_before_after("results_v1", "results")
    df_ba.to_csv("results/before_after.csv", index=False)

    stats = extract_normalisation_stats("results/normalisation_stats.csv")
    stats["cl1_files_used"] = extract_cl1_files_used("src/cl1.py")

    lid_st, he_map = extract_lid_stats("results")
    stats.update(lid_st)
    stats.update(extract_error_summary_stats("results/lid_error_summary.csv"))

    base_st, best_cfg = extract_baseline_stats("results")
    stats.update(base_st)

    pd.DataFrame(list(stats.items()), columns=["name", "value"]).to_csv("results/derived_stats.csv", index=False)

    tables = load_tables("results", best_cfg, he_map)
    notes = render_report(template, stats, tables)
    with open("NOTES.md", "w", encoding="utf-8") as f:
        f.write(notes)

    with open("results/summary.md", "w", encoding="utf-8") as f:
        f.write("\n\n".join(f"## {name}\n\n{content}" for name, content in tables.items()) + "\n")


if __name__ == "__main__":
    main()
