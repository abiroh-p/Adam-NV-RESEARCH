# Manuscript vs recomputed values

Source manuscript: `ResearchPaper_Revised/IEEE_Access_Result_Revised.docx` (text extracted with pandoc). **The manuscript was not edited.** This report lists where its statements disagree with the raw logs.

Calculation method (all E13-E15 rows): per-seed mean validation accuracy over epochs 56-60 (`--metric last5`); paired by seed; mean difference with t-based 95% CI; paired t-test; exact two-sided sign-flip test; Holm adjustment within each experiment family exactly as coded in `analysis/analyze_e13_e15.py` (code unchanged apart from strict loading and the added `--metric` switch; numeric output identical to the unmodified script on the same logs). Differences are in fractions of accuracy unless a column says `percentage_points`.

## M1 — Section 4.1 (E1), Figure 4 caption and table
> "The repository contains four Adam-NV runs but only two complete runs for Adam, AdamW, and AdaBelief; inferential E1 comparisons are therefore withheld pending completion of baseline seeds 2 and 3."

**Repository fact:** seeds 0-3 exist for all four optimizers (16 CSVs in `e1_convergence/logs`), seeds 4-9 for Adam/AdamW/Adam-NV, GONO seeds 0-9. The table values below do not match any seed subset (last-5-epoch mean; SD with ddof=1):

| optimizer | manuscript_mean | manuscript_sd | manuscript_n | recomputed_subset | n | recomputed_last5_mean | recomputed_sd_ddof1 |
|---|---|---|---|---|---|---|---|
| adam_nv | 0.9337 | 0.002 | 4 | seeds 0-3 | 4 | 0.9327 | 0.0018 |
| adam_nv | 0.9337 | 0.002 | 4 | all available seeds | 10 | 0.932 | 0.0015 |
| adam | 0.9319 | 0.0012 | 2 | seeds 0-3 | 4 | 0.9315 | 0.0015 |
| adam | 0.9319 | 0.0012 | 2 | seeds 0-1 | 2 | 0.9315 | 0.0011 |
| adam | 0.9319 | 0.0012 | 2 | all available seeds | 10 | 0.9315 | 0.0011 |
| adamw | 0.9314 | 0.0017 | 2 | seeds 0-3 | 4 | 0.9314 | 0.0009 |
| adamw | 0.9314 | 0.0017 | 2 | seeds 0-1 | 2 | 0.9313 | 0.0015 |
| adamw | 0.9314 | 0.0017 | 2 | all available seeds | 10 | 0.9322 | 0.0015 |
| adabelief | 0.9267 | 0.0 | 2 | seeds 0-3 | 4 | 0.9248 | 0.0021 |
| adabelief | 0.9267 | 0.0 | 2 | seeds 0-1 | 2 | 0.9253 | 0.0002 |
| adabelief | 0.9267 | 0.0 | 2 | all available seeds | 4 | 0.9248 | 0.0021 |

GONO now has all 10 seeds (seed 1 via the author's verified rerun; identical to the stdout-recovery route: last-5 mean 0.92445). Recomputed E1 summaries and paired contrasts: `analysis/results_recomputed/e1_summary_*.csv`, `e1_paired_*.csv`. Adam-NV minus Adam (10 seeds, last-5): +0.059 pp, 95% CI [-0.05, +0.16] pp, paired t p = 0.238, sign-flip p = 0.231. The Holm family for E1 is *proposed* in the script and needs author confirmation (register STA-06).

## M2 — Section 4.9, E14/E15 intermittent beta2 = 0.95
> "One intermittent E14 run ends at epoch 32, so the analysis correctly refuses publication-mode output."

**Repository fact:** `response_lag/..._beta20.95_seed1.csv` has 60 contiguous epochs; publication-mode analysis succeeds (22/22 comparisons complete). The run has no metadata sidecar, so its provenance cannot be confirmed (register LOG-02). Values that change when the stale table is replaced:

| experiment | comparison | quantity | original_committed | corrected | reason |
|---|---|---|---|---|---|
| E14 | ramp_up: coupled beta2=0.99 minus coupled beta2=0.999 | p_paired_t_holm | 0.733967 | 0.656672 | Holm family recomputed after raw p-values changed |
| E14 | intermittent: coupled beta2=0.99 minus coupled beta2=0.999 | p_paired_t_holm | 0.733967 | 0.638775 | Holm family recomputed after raw p-values changed |
| E14 | intermittent: coupled beta2=0.95 minus coupled beta2=0.999 | n_pairs | 3 | 4 | original table built with 3 of 4 paired seeds (seed 1 was truncated) |
| E14 | intermittent: coupled beta2=0.95 minus coupled beta2=0.999 | mean_difference_percentage_points | 0.467333 | 0.438 | original table built with 3 of 4 paired seeds (seed 1 was truncated) |
| E14 | intermittent: coupled beta2=0.95 minus coupled beta2=0.999 | ci95_low | -0.00240717 | 0.000560976 | original table built with 3 of 4 paired seeds (seed 1 was truncated) |
| E14 | intermittent: coupled beta2=0.95 minus coupled beta2=0.999 | ci95_high | 0.0117538 | 0.00819902 | original table built with 3 of 4 paired seeds (seed 1 was truncated) |
| E14 | intermittent: coupled beta2=0.95 minus coupled beta2=0.999 | p_paired_t_raw | 0.104852 | 0.0354944 | original table built with 3 of 4 paired seeds (seed 1 was truncated) |
| E14 | intermittent: coupled beta2=0.95 minus coupled beta2=0.999 | p_sign_flip_exact | 0.25 | 0.125 | original table built with 3 of 4 paired seeds (seed 1 was truncated) |
| E14 | intermittent: coupled beta2=0.95 minus coupled beta2=0.999 | p_paired_t_holm | 0.733967 | 0.248461 | original table built with 3 of 4 paired seeds (seed 1 was truncated) |
| E15 | intermittent: decoupled beta2_noise=0.95 minus coupled beta2=0.95 | n_pairs | 3 | 4 | original table built with 3 of 4 paired seeds (seed 1 was truncated) |
| E15 | intermittent: decoupled beta2_noise=0.95 minus coupled beta2=0.95 | mean_difference_percentage_points | -0.199333 | -0.228 | original table built with 3 of 4 paired seeds (seed 1 was truncated) |
| E15 | intermittent: decoupled beta2_noise=0.95 minus coupled beta2=0.95 | ci95_low | -0.00830467 | -0.00570464 | original table built with 3 of 4 paired seeds (seed 1 was truncated) |
| E15 | intermittent: decoupled beta2_noise=0.95 minus coupled beta2=0.95 | ci95_high | 0.00431801 | 0.00114464 | original table built with 3 of 4 paired seeds (seed 1 was truncated) |
| E15 | intermittent: decoupled beta2_noise=0.95 minus coupled beta2=0.95 | p_paired_t_raw | 0.307129 | 0.124329 | original table built with 3 of 4 paired seeds (seed 1 was truncated) |
| E15 | intermittent: decoupled beta2_noise=0.95 minus coupled beta2=0.95 | p_sign_flip_exact | 0.5 | 0.25 | original table built with 3 of 4 paired seeds (seed 1 was truncated) |
| E15 | intermittent: decoupled beta2_noise=0.95 minus coupled beta2=0.95 | p_paired_t_holm | 1 | 0.621647 | original table built with 3 of 4 paired seeds (seed 1 was truncated) |

## M3 — Section 4.9, "no E13 contrast and no E15 contrast survives Holm correction"
Holds under the **last-5-epoch** metric (min Holm p: E13 0.507, E13 interaction 0.272, E15 0.068). Under the **final-epoch** metric, one E15 contrast crosses 0.05:

| experiment | comparison | mean_difference_percentage_points_last5 | mean_difference_percentage_points_final | p_paired_t_holm_last5 | p_paired_t_holm_final |
|---|---|---|---|---|---|
| E14 | ramp_up: coupled beta2=0.95 minus coupled beta2=0.999 | -0.161 | -0.125 | 0.0279 | 0.6793 |
| E15 | ramp_up: decoupled beta2_noise=0.95 minus coupled beta2=0.95 | 0.162 | 0.1475 | 0.2761 | 0.004593 |

## M4 — Section 4.9, "E14 suggests a ramp-up degradation at beta2=0.95 ... sign-flip p-value is 0.125"
Holds under last-5 (Holm 0.028, sign-flip 0.125); under final-epoch the Holm p is 0.679 (table above). The choice of metric is an author decision (register STA-03).

## M5 — Section 5.7 / discussion text
"The available E1 repository logs are incomplete for the baseline optimizers" — no longer true of the repository snapshot except for AdaBelief seeds 4-9 (only 4 AdaBelief seeds).

## Newer paper draft
`adam-nv/paperdraft/adamnv_paper_draft.pdf` supersedes the docx compared above; it was recomputed claim by claim in `PAPER_DRAFT_CHECK.md` (78 match, 6 not verifiable, 1 inconsistent). Sections M1-M5 describe the older `IEEE_Access_Result_Revised.docx` only.

## Not checked
E2, E7, E9, E11, E12 tables and figures were not compared with the logs in this repair.
