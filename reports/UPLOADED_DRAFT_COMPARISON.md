# Uploaded paper draft vs the draft already in the repository

Files compared (both LaTeX/hyperref PDFs, neither edited):

| file | built (PDF metadata) | pages |
|---|---|---|
| `adam-nv/paperdraft/adamnv_paper_draft_uploaded_20261026T0658Z.pdf` (the file you uploaded; SHA-256 `bac213d8…`) | 2026-10-26 06:58 UTC | 7 |
| `adam-nv/paperdraft/adamnv_paper_draft.pdf` (pushed in commit `c534451`; SHA-256 `d4cf6d6c…`) | 2026-10-26 22:41 UTC | 8 |

**The uploaded file is the older version.** I kept both so the history is not lost.

## What differs between the versions (number-level and sentence-level diff)
Only in the **uploaded (older)** version: the first timing benchmark (optimizer step 31.0 ms vs 4.2 ms, 7.4x; end-to-end 127.4 vs 105.5 ms; +20.8% by mean, +20.3% by median; forward/backward not isolated; "repeated ten times"; AdamW 106.3 ms, AdaBelief 110.1 ms; state-extra 85.2 MiB), the abstract's "about 20%", a label-noise table without the 0.92 / 0.95 constants or the shuffle rows, and four TODOs.
Only in the **newer** version: timing v2 (20 rounds, randomized order, medians, paired ratios: +17.1%), Experiments 11 and 12 (shuffle control; constant beta1 0.92 / 0.95), the element-level beta1 distribution table, and the corresponding abstract, discussion and conclusion text. Every other number I compared is identical in both.

## Claims
All claims the two versions share (clean benchmark, learning-rate and constant-beta1 tables, noise tables, controller direction, CIFAR-100, label-noise and dose-response tables, GONO variant, logged beta1 statistics, epochs-to-90% / validation-loss AUC) were recomputed from the raw logs: see `PAPER_DRAFT_CHECK.md`. Current totals over all claims of both versions: {'MATCH': 95, 'NOT VERIFIABLE': 6, 'DIFFERS': 1}.

Claims that exist **only** in the uploaded version are the v1 timing numbers. **They are NOT VERIFIABLE from the repository**: it contains no v1 timing results, only `timing_v2_results.zip`. The v2 results give +17.1% end-to-end (CI [16.9, 17.5]%), optimizer step 27.3 vs 3.8 ms, forward/backward 99.0 vs 104.2 ms; the difference from v1 (+20.8%) is the methodological change the newer draft describes (randomized order, medians).

## The uploaded draft's TODOs, against the repository
| TODO in the uploaded draft | status |
|---|---|
| Repeat the timing with the interleaved benchmark and report peak GPU memory | Timing v2 results are in the repository and used by the newer draft. The v2 summary also contains peak allocated memory (Adam 828.5 MiB; Adam-NV 914.1 MiB, +85.6 MiB), which neither draft reports. |
| Report the element-level beta1 distribution from the shuffle-control runs | Done in the newer draft (Table VII). It reproduces exactly from `beta1_dist_none_seed{0,1}.jsonl` in `exp11_logs_C.zip` (the 'none' arm, 11,173,962 elements). The 'none' arm is also bit-identical to the original Adam-NV label-noise runs. |
| Clean the repository layout and reconcile the stored source files with the hashes recorded in the logs before the link is published | **Partly achievable, not fully.** Done here: `provenance/` archives every recoverable source version byte-exactly and `tools/provenance.py verify` checks each recorded hash. Not possible: 48 runs (28 E1 runs incl. all GONO seeds, 20 EXP03 runs) recorded hashes for source files that exist nowhere in the repository, git history or zips; 96 runs recorded no hash at all. The sentence "Logs record ... SHA-256 hashes of the source files" is therefore true only for part of the logs, and "reconcile" can only mean *document* those runs as unverified, which `reports/DISCREPANCY_REGISTER.md` (PRV-01, PRV-02, PRV-04) does. Behavioural evidence: the GONO seed-1 rerun with fully verified hashes reproduced the original run bit-for-bit over 59 epochs. |
| Confirm acknowledgment wording and funding / compute statement | Not something the repository can answer. |

## Sentences worth re-reading before publication (factual, not stylistic)
* "Logs record library versions, arguments and SHA-256 hashes of the source files" - see the row above.
* "a repeat of two label-noise runs with the final code reproduced the earlier logs exactly" - supported: the EXP07 canary re-run of seed 0 (Adam and Adam-NV) differs by 0.0 in every logged column.
* Table I (both versions list 10 repeats for the overhead benchmark): consistent with the *uploaded* draft's v1 timing text, but inconsistent with the *newer* draft's own Section VII (20 rounds).
* AdaBelief is reported over 10 seeds in both versions; the repository contains 4 AdaBelief seeds. The constant-beta1 0.83 / 0.80 ten-seed contrasts need the Exp. 6 logs, which are also missing.
