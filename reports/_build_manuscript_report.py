import pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parent
def md(df, fmt=None):
    cols = list(df.columns); fmt = fmt or {}
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join((fmt.get(c, lambda v: str(v)))(r[c]) for c in cols) + " |")
    return "\n".join(out)
g = lambda v: f"{v:.6g}"
chg = pd.read_csv(R / "e13_e15_original_vs_corrected.csv")
e1 = pd.read_csv(R / "e1_manuscript_vs_recomputed.csv")
sens = pd.read_csv(R / "metric_definition_sensitivity.csv")
t = ["# Manuscript vs recomputed values",
"",
"Source manuscript: `ResearchPaper_Revised/IEEE_Access_Result_Revised.docx` (text extracted with pandoc). **The manuscript was not edited.** This report lists where its statements disagree with the raw logs.",
"",
"Calculation method (all E13-E15 rows): per-seed mean validation accuracy over epochs 56-60 (`--metric last5`); paired by seed; mean difference with t-based 95% CI; paired t-test; exact two-sided sign-flip test; Holm adjustment within each experiment family exactly as coded in `analysis/analyze_e13_e15.py` (code unchanged apart from strict loading and the added `--metric` switch; numeric output identical to the unmodified script on the same logs). Differences are in fractions of accuracy unless a column says `percentage_points`.",
"",
"## M1 — Section 4.1 (E1), Figure 4 caption and table",
"> \"The repository contains four Adam-NV runs but only two complete runs for Adam, AdamW, and AdaBelief; inferential E1 comparisons are therefore withheld pending completion of baseline seeds 2 and 3.\"",
"",
"**Repository fact:** seeds 0-3 exist for all four optimizers (16 CSVs in `e1_convergence/logs`), seeds 4-9 for Adam/AdamW/Adam-NV, GONO seeds 0-9. The table values below do not match any seed subset (last-5-epoch mean; SD with ddof=1):",
"", md(e1), "",
"GONO now has all 10 seeds (seed 1 via the author's verified rerun; identical to the stdout-recovery route: last-5 mean 0.92445). Recomputed E1 summaries and paired contrasts: `analysis/results_recomputed/e1_summary_*.csv`, `e1_paired_*.csv`. Adam-NV minus Adam (10 seeds, last-5): +0.059 pp, 95% CI [-0.05, +0.16] pp, paired t p = 0.238, sign-flip p = 0.231. The Holm family for E1 is *proposed* in the script and needs author confirmation (register STA-06).",
"",
"## M2 — Section 4.9, E14/E15 intermittent beta2 = 0.95",
"> \"One intermittent E14 run ends at epoch 32, so the analysis correctly refuses publication-mode output.\"",
"",
"**Repository fact:** `response_lag/..._beta20.95_seed1.csv` has 60 contiguous epochs; publication-mode analysis succeeds (22/22 comparisons complete). The run has no metadata sidecar, so its provenance cannot be confirmed (register LOG-02). Values that change when the stale table is replaced:",
"", md(chg, {"original_committed": g, "corrected": g}), "",
"## M3 — Section 4.9, \"no E13 contrast and no E15 contrast survives Holm correction\"",
"Holds under the **last-5-epoch** metric (min Holm p: E13 0.507, E13 interaction 0.272, E15 0.068). Under the **final-epoch** metric, one E15 contrast crosses 0.05:",
"", md(sens, {c: (lambda v: f"{v:.4g}") for c in sens.columns[2:]}), "",
"## M4 — Section 4.9, \"E14 suggests a ramp-up degradation at beta2=0.95 ... sign-flip p-value is 0.125\"",
"Holds under last-5 (Holm 0.028, sign-flip 0.125); under final-epoch the Holm p is 0.679 (table above). The choice of metric is an author decision (register STA-03).",
"",
"## M5 — Section 5.7 / discussion text",
"\"The available E1 repository logs are incomplete for the baseline optimizers\" — no longer true of the repository snapshot except for AdaBelief seeds 4-9 (only 4 AdaBelief seeds).",
"",
"## Newer paper draft\n`adam-nv/paperdraft/adamnv_paper_draft.pdf` supersedes the docx compared above; it was recomputed claim by claim in `PAPER_DRAFT_CHECK.md` (78 match, 6 not verifiable, 1 inconsistent). Sections M1-M5 describe the older `IEEE_Access_Result_Revised.docx` only.\n\n## Not checked",
"E2, E7, E9, E11, E12 tables and figures were not compared with the logs in this repair.",
""]
(R / "MANUSCRIPT_DISCREPANCIES.md").write_text("\n".join(t), encoding="utf-8")
print("ok")
