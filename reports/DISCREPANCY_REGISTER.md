# Discrepancy register

Audit date 2026-10-25. Evidence commands and outputs are in `reports/` (`provenance_verification.csv`, `run_completeness.csv`, `*_original_vs_corrected.csv`, `validation_logs/`).
Status vocabulary: **RESOLVED** (fixed and checked), **PARTIALLY RESOLVED**, **MITIGATED** (risk reduced, cause remains), **DOCUMENTED** (cannot be fixed; recorded), **UNVERIFIED**, **UNRESOLVED**, **NEEDS AUTHOR DECISION**, **OPEN** (change outside the repo, e.g. manuscript).

| ID | Severity | Final status |
|---|---|---|
| PRV-01 | critical | UNRESOLVED (provenance unverified) |
| PRV-02 | high | UNRESOLVED (partial provenance) |
| PRV-03 | low | RESOLVED (bytes verified; naming history undocumented) |
| PRV-04 | high | DOCUMENTED (unverifiable) |
| PRV-05 | high | RESOLVED |
| PRV-06 | medium | DOCUMENTED (equivalence with lost version unverified) |
| PRV-07 | medium | RESOLVED |
| LOG-01 | high | RESOLVED (by author-supplied rerun) |
| LOG-02 | medium | RESOLVED (documentation) / UNVERIFIED (run provenance) |
| LOG-03 | low | RESOLVED |
| LOG-04 | low | MITIGATED |
| LOG-05 | high | UNRESOLVED |
| LOG-06 | high | RESOLVED (by author-supplied EXP09 data) |
| STA-01 | high | RESOLVED |
| STA-02 | high | RESOLVED (code) / NEEDS AUTHOR DECISION (which metric governs) |
| STA-03 | high | NEEDS AUTHOR DECISION |
| STA-04 | high | OPEN (manuscript edit required by authors) |
| STA-05 | medium | RESOLVED (new table) / original retained |
| STA-06 | medium | NEEDS AUTHOR DECISION |
| STA-07 | low | UNVERIFIED |
| STA-08 | medium | RESOLVED |
| OBS-01 | medium | RESOLVED (paths/scripts) ; lost scripts' numbers not reproducible |
| OBS-02 | medium | UNRESOLVED (documents not recoverable) |
| OPT-01 | medium | DOCUMENTED (docstring overstated) |
| OPT-02 | low | UNVERIFIED |
| DOC-01 | medium | RESOLVED |
| DOC-02 | low | MITIGATED |
| DAT-01 | high | PARTIALLY RESOLVED (Exp.6 control logs still missing) |
| DAT-02 | low | RESOLVED (preserved) |
| DAT-03 | low | RESOLVED (by author-supplied EXP11 logs) |
| DAT-04 | low | RESOLVED (data verified) |
| DAT-05 | low | UNVERIFIED (hardware-specific; summary consistent) |
| DOC-03 | medium | OPEN (author edit of Table I; 6 claims need missing logs) |

## PRV-01 — Source-code or configuration mismatch — severity: critical

- **Affected files:** adam-nv/e1_convergence/logs/gono_seed{0-9}_metadata.json; adam-nv/e1_convergence/AdditionalsLogs/{adam,adamw,adam_nv}_seed{4-9}_metadata.json (+ copies in adam-nv.zip)
- **Experiment / optimizer / seed:** E1 protocol v2: GONO seeds 0-9; Adam, AdamW, Adam-NV seeds 4-9 (28 runs)
- **Evidence:** Recorded code_sha256 for adam_nv.py (96971286…), gono.py (82665f3c…) and train_resnet.py (7b5134fd…) match NO source in the worktree, all 7 git commits, or any nested zip, also not under LF/CRLF/BOM variants (tools/provenance.py verify; /reports/provenance_verification.csv). experiment_metadata.py hash (92864b04…) does verify. NEW EVIDENCE (push ba29996): the author's GONO seed-1 rerun, whose hashes all verify (adam_nv bcc7824b, gono 326bcedc, train_resnet 2ab3aa02), reproduces the original seed-1 log bit-for-bit over epochs 1-59 (12 non-timing columns, max abs diff 0.0) although the original run recorded the unarchived hashes. This is behavioural evidence that the lost versions behaved identically for GONO seed 1 on that stack; it does not verify any hash and covers one seed only.
- **Corrective action:** Cannot be repaired from available evidence. Provenance for these runs is marked UNVERIFIED_NOT_ARCHIVED; no newer file was associated with them. Verifier and check_logs.py now report this explicitly.
- **Requires experiment rerun:** Would need the original Kaggle source files; rerun alone would not restore provenance of the existing logs
- **Validation:** tools/provenance.py verify lists 28 runs x 3 files UNVERIFIED; tests test_known_unverified_hashes_are_still_reported_unverified, test_verify_walks_directories_and_nested_zips...
- **Final status:** **UNRESOLVED (provenance unverified)**

## PRV-02 — Source-code or configuration mismatch — severity: high

- **Affected files:** adam-nv/e1_convergence/exp03_logs_adamnv.zip; exp03_logs_gono.zip
- **Experiment / optimizer / seed:** EXP03: adam_nv_basecorr seeds 0-9 and gono_exact seeds 0-9 (20 runs)
- **Evidence:** train_resnet.py recorded as 5ac384a1…, not preserved anywhere. adam_nv.py (bcc7824b) and gono.py (326bcedc) of the same runs DO verify.
- **Corrective action:** Marked unverified for train_resnet.py only. The EXP03 analysis script and PROTOCOL_V2.md are also missing (see OBS-02).
- **Requires experiment rerun:** Original train_resnet.py needed
- **Validation:** provenance verify: 20 runs UNVERIFIED for train_resnet.py only
- **Final status:** **UNRESOLVED (partial provenance)**

## PRV-03 — Missing or inconsistent metadata — severity: low

- **Affected files:** adam-nv/e1_convergence/exp05_logs_{adam,adamnv,gono}.zip metadata
- **Experiment / optimizer / seed:** EXP05 LR sensitivity (24 runs)
- **Evidence:** experiment_metadata.py hashes the file literally named train_resnet.py (_HASHED_FILES). The exp05 runs recorded 8aef0072…, which equals the bytes the repository stores only as train_resnet_ex.py; the repository's own train_resnet.py (2ab3aa02…) was referenced by no run at audit time; since push ba29996 it is recorded by the GONO seed-1 rerun and by EXP10 and is archived as train_resnet.py@2ab3aa02. No script in the repo copies/renames train_resnet_ex.py, so how the file came to carry that name at run time is not documented. Push ba29996 DELETED train_resnet_ex.py from the working tree; its bytes remain in git history, in the newer adam-nv.zip and in provenance/source_archive, and the repaired tree keeps the file.
- **Corrective action:** Archived the exact bytes as provenance/source_archive/train_resnet_ex.py@8aef0072; verifier reports VERIFIED_EXACT_RENAMED (bytes verified, name differs) rather than EXACT. The bytes are verified; the naming history is not.
- **Requires experiment rerun:** No
- **Validation:** provenance verify: 48 claims (24 runs x 2 copies) VERIFIED_EXACT_RENAMED
- **Final status:** **RESOLVED (bytes verified; naming history undocumented)**

## PRV-04 — Missing or inconsistent metadata — severity: high

- **Affected files:** e1_convergence/logs/*_metadata.json (16 runs); e2_noisy_gradient/logs/{Label_Diagnostic,LagDiagnostic,cifar100_generality,controller_direction}
- **Experiment / optimizer / seed:** E1 seeds 0-3 (all optimizers); E2 label diagnostics, lag, CIFAR-100, controller direction
- **Evidence:** 96 runs have neither a per-run hash nor a folder-level _code_hashes file: E1 seeds 0-3 (16), Label_Diagnostic (8), LagDiagnostic (24), cifar100_generality (24), controller_direction (24). EXP07/08/09 runs have no per-run hash but are covered by verified folder-level _code_hashes files (verifier verdict COVERED_BY_FOLDER_HASHES; weaker than a per-run hash). Separately, 209 per-epoch runs have no metadata sidecar (E12 24, Ablation 38, Intermittent 16, decoupling 32, diagnose 7, dynamic_noise 28, part_b 16, part_c 16, response_lag 32). Behavioural evidence exists only for Label_Diagnostic seed 0 (EXP07 canary re-run: bit-identical, Adam and Adam-NV).
- **Corrective action:** Cannot be recreated. Reported as NO_HASH_RECORDED / metadata=NO. Behavioural evidence only for Label_Diagnostic seed 0: the EXP07 canary re-ran seed 0 with hashed code and matched the original logs bit-for-bit (0.0 difference, Adam and Adam-NV). Seeds 1-3 are not covered by the canary.
- **Requires experiment rerun:** No (cannot restore)
- **Validation:** analyze_exp07.py canary output; tools/check_runs.py
- **Final status:** **DOCUMENTED (unverifiable)**

## PRV-05 — Obsolete or broken script — severity: high

- **Affected files:** adam-nv/e1_convergence/src/check_logs.py
- **Experiment / optimizer / seed:** All runs checked with this script
- **Evidence:** The 'hash check' only tested that the code_sha256 field existed (line 63); it never compared the hash with any file.
- **Corrective action:** Patched: compares every recorded hash with the registry (strict/warn/off). New tools/provenance.py scans metadata and _code_hashes_*.txt in directories and nested zips.
- **Requires experiment rerun:** No
- **Validation:** tests test_check_logs_strict_fails_on_unarchived_hash_and_warn_passes, test_check_logs_strict_passes_a_run_whose_hashes_are_archived, ProvenanceToolTests (6 tests)
- **Final status:** **RESOLVED**

## PRV-06 — Source-code or configuration mismatch — severity: medium

- **Affected files:** adam-nv/e1_convergence/src/adam_nv.py; e2_noisy_gradient/src/adam_nv.py; adam-nv/EXP01_RUNBOOK.md
- **Experiment / optimizer / seed:** E1 / E2 Adam-NV
- **Evidence:** Two optimizer versions: 2d6cd7df (no bias_correction option) and bcc7824b. EXP01 runbook says the E1 adam_nv.py 'produced seeds 0-3' and must not be replaced, but nested E1 adam_nv.py is byte-identical to bcc7824b, while the 28 PRV-01 runs recorded 96971286.
- **Corrective action:** Archived both versions. New test shows 2d6cd7df and bcc7824b are bit-identical over 250 steps at default, smooth, decoupled and gamma=0 settings. This supports, but does NOT prove, equivalence with the lost 96971286 version.
- **Requires experiment rerun:** No
- **Validation:** tests/test_repair_optimizer.py::test_old_and_new_versions_are_bit_identical_at_default_settings
- **Final status:** **DOCUMENTED (equivalence with lost version unverified)**

## PRV-07 — Missing or inconsistent metadata — severity: medium

- **Affected files:** adam-nv/e2_noisy_gradient/src/noise.py; EXP08_RUNBOOK.md; adam-nv.zip
- **Experiment / optimizer / seed:** EXP07/EXP08 (all runs)
- **Evidence:** Pinned hash eee61201 differs from worktree noise.py (ca589851). The copy in adam-nv.zip has mixed line endings and matches the pin byte-for-byte; contents are identical after LF normalization. No .gitattributes existed, so git can silently change hashes.
- **Corrective action:** Archived the exact historical bytes (noise.py@eee61201); added .gitattributes (-text for sources/logs) to stop line-ending rewrites; verifier distinguishes EXACT from EOL_NORMALIZED.
- **Requires experiment rerun:** No
- **Validation:** provenance verify: 13 noise.py claims VERIFIED_EXACT; test_line_ending_variant_is_eol_normalized_not_exact
- **Final status:** **RESOLVED**

## LOG-01 — Missing, truncated, duplicated, or incomplete logs — severity: high

- **Affected files:** adam-nv/e1_convergence/logs/gono_seed1.csv (and copy in adam-nv.zip)
- **Experiment / optimizer / seed:** E1 GONO seed 1
- **Evidence:** CSV ends at epoch 59. Suite status: success, 3029.5 s. stdout_gono_gpu0.txt line 127 contains the epoch-60 line (train_loss 0.0037, train_acc 0.9989, val_loss 0.4386, val_acc 0.9256); all 59 CSV epochs match stdout (0 mismatches). UPDATE (push ba29996): the author reran GONO seed 1 (adam-nv/e1_convergence/logs/gono_seed_1/, deterministic, all four hashes verify). The rerun has 60 epochs; epochs 1-59 equal the original in all 12 non-timing columns (max abs diff 0.0); epoch 60 (val_acc 0.9256, train_acc 0.99892, val_loss 0.438588, train_loss 0.003655) equals the stdout-recovered values at 4 dp.
- **Corrective action:** Complete rerun adopted by analysis/analyze_e1.py ONLY after a per-run consistency check (original kept, registered as KNOWN_PARTIAL_SUPERSEDED, both copies); the stdout supplement is retained as an independently corroborated record. E1 GONO now has 10 seeds. Last-5 mean via rerun (0.92445) equals the supplement route.
- **Requires experiment rerun:** Done by the author, not by the repair
- **Validation:** analysis/results_recomputed/e1_rerun_consistency.csv; tests ZipScanAndRerunPolicyTests (consistent merge, refusal on inconsistency, original wins); validate_all steps 17b, 18
- **Final status:** **RESOLVED (by author-supplied rerun)**

## LOG-02 — Documentation inconsistency — severity: medium

- **Affected files:** README.md; RESEARCH_STATUS.md; ResearchPaper_Revised/IEEE_Access_Result_Revised.docx (section 4.9); e2_noisy_gradient/logs/response_lag/adam_nv_scheduleintermittent_sigma2.0_beta20.95_seed1.csv
- **Experiment / optimizer / seed:** E14/E15 intermittent beta2=0.95, Adam-NV seed 1
- **Evidence:** Docs say the run stops at epoch 32. The file has 60 contiguous epochs and has been in the repo since the initial commit. response_lag/ contains no metadata sidecars at all, so it cannot be shown to be the original run.
- **Corrective action:** Docs corrected (README, RESEARCH_STATUS addendum). Recomputed statistics use the complete file. The manuscript (docx) was NOT edited; see reports/MANUSCRIPT_DISCREPANCIES.md.
- **Requires experiment rerun:** No (rerun would replace, not verify, the evidence)
- **Validation:** check_runs: status OK; analyze_e13_e15 publication mode succeeds
- **Final status:** **RESOLVED (documentation) / UNVERIFIED (run provenance)**

## LOG-03 — Missing, truncated, duplicated, or incomplete logs — severity: low

- **Affected files:** e2_noisy_gradient/logs/dynamic_noise/adam_nv_scheduleintermittent_sigma2.0_beta20.95_seed3.csv (14 rows)
- **Experiment / optimizer / seed:** E14 intermittent beta2=0.95 seed 3
- **Evidence:** First 14 rows equal the first 14 rows of the complete seed-3 run in every column except epoch_time_sec; the '(1)' copy is byte-identical to response_lag/..seed3.csv.
- **Corrective action:** File kept; registered in reports/known_superseded_logs.json with its SHA-256; check_runs reports KNOWN_PARTIAL_SUPERSEDED. Analyses read response_lag/.
- **Requires experiment rerun:** No
- **Validation:** tools/check_runs.py
- **Final status:** **RESOLVED**

## LOG-04 — Missing, truncated, duplicated, or incomplete logs — severity: low

- **Affected files:** e2_noisy_gradient/logs/{Intermittent,dynamic_noise,controller_direction,response_lag}
- **Experiment / optimizer / seed:** E13/E14
- **Evidence:** Intermittent/ duplicates dynamic_noise/ byte-for-byte; controller_direction/ reuses file names with different content. Analyses keyed by file name alone would mix them.
- **Corrective action:** check_runs annotates duplicates and same-name-different-content files; analyses use explicit full paths.
- **Requires experiment rerun:** No
- **Validation:** tools/check_runs.py output
- **Final status:** **MITIGATED**

## LOG-05 — Missing, truncated, duplicated, or incomplete logs — severity: high

- **Affected files:** adam-nv/e1_convergence/ (AdaBelief) 
- **Experiment / optimizer / seed:** E1 AdaBelief
- **Evidence:** AdaBelief has seeds 0-3 only; Adam/AdamW/Adam-NV/GONO have 10. Paired comparison vs AdaBelief is limited to n=4 (sign-flip minimum p = 0.125). The EXP10 runbook refers to an 'AdaBelief extension (hash 2ab3aa02)', which implies such runs exist, but no AdaBelief seed >=4 log is anywhere in the repository (checked loose files and all zips). The paper draft (Table II, section V.A) reports AdaBelief over 10 seeds (92.44 +/- 0.17; -0.71 pp vs Adam), so those logs exist outside the repository.
- **Corrective action:** Command documented in README (adabelief seeds 4-9). Not run.
- **Requires experiment rerun:** Yes (GPU)
- **Validation:** analyze_e1.py reports n_pairs=4
- **Final status:** **UNRESOLVED**

## LOG-06 — Missing, truncated, duplicated, or incomplete logs — severity: high

- **Affected files:** e2_noisy_gradient/src/run_controller_direction_suite.py; logs/controller_direction
- **Experiment / optimizer / seed:** Smooth vs distrust under label noise
- **Evidence:** UPDATE (push ba29996): Experiment 9 supplies the missing arm: 8 complete seeds (0-7) of Adam-NV --response-mode smooth, 20% label noise, deterministic, in adam-nv/e2_noisy_gradient/logs/exp09_logs_{A,B}.zip. All five folder-level hashes (242e6a77, bcc7824b, 90e11e2c, eee61201, ab020aaf) verify against the archived sources; all runs have 60 contiguous epochs and metadata. Re-analysis with analyze_exp09.py: M = +3.91 pp, F (smooth - Adam) = -1.47 pp, D (distrust - smooth) = +5.38 pp, CI [+4.55, +6.21]; the script's pre-registered outcome: DISTRUST DIRECTION DRIVES THE EFFECT. The file names predicted by the repair's own suite matched the real EXP09 outputs.
- **Corrective action:** No repair action needed; the data came from the author. The repair's run_label_noise_direction_suite.py is marked SUPERSEDED. Re-analysis recorded in validation step 13c.
- **Requires experiment rerun:** Done by the author, not by the repair
- **Validation:** check_runs (zip scan): all 8 OK; provenance verify: folder hashes VERIFIED_EXACT; analyze_exp09.py output
- **Final status:** **RESOLVED (by author-supplied EXP09 data)**

## STA-01 — Statistical-analysis inconsistency — severity: high

- **Affected files:** analysis/results/e13_e15_primary_statistics.csv
- **Experiment / optimizer / seed:** E13-E15
- **Evidence:** Committed table is stale: built with 3 paired seeds (n_pairs=3, complete=False) for intermittent beta2=0.95 contrasts. Recomputation from logs with the unchanged analysis code gives 22/22 complete comparisons.
- **Corrective action:** Original preserved. Corrected tables in analysis/results_recomputed/. 16 values in 4 comparisons changed (reports/e13_e15_original_vs_corrected.csv).
- **Requires experiment rerun:** No
- **Validation:** test_committed_recomputed_tables_match_a_fresh_recomputation
- **Final status:** **RESOLVED**

## STA-02 — Statistical-analysis inconsistency — severity: high

- **Affected files:** analysis/analyze_e13_e15.py; adam-nv/analysis/analyze_exp06.py; e1_results_table.csv
- **Experiment / optimizer / seed:** All accuracy summaries
- **Evidence:** Final-epoch vs last-5-epoch accuracy were not distinguished in the analysis outputs (metric label 'trailing_5_epoch'; e1_results_table.csv column final_val_acc; analyze_exp06 uses final-epoch while E13-E15 use last-5).
- **Corrective action:** run_io.py defines both metrics as separate functions; analyze_e13_e15.py has --metric last5|final with different output file names and metric labels; analyze_e1.py reports both. analyze_exp06/07/08 left unchanged (they state their metric).
- **Requires experiment rerun:** No
- **Validation:** test_final_epoch_and_last5_are_different_quantities; test_metric_switch_changes_the_computed_quantity
- **Final status:** **RESOLVED (code) / NEEDS AUTHOR DECISION (which metric governs)**

## STA-03 — Statistical-analysis inconsistency — severity: high

- **Affected files:** analysis/results_recomputed/e13_e15_{last5,final}_statistics.csv
- **Experiment / optimizer / seed:** E14 ramp-up beta2=0.95; E15 ramp-up decoupled 0.95
- **Evidence:** Holm significance flips with the metric definition: E14 ramp-up 0.95: Holm p 0.028 (last-5) vs 0.679 (final); E15 ramp-up decoupled 0.95: 0.276 (last-5) vs 0.005 (final).
- **Corrective action:** Both reported side by side (reports/metric_definition_sensitivity.csv). No conclusion was chosen, per the instruction to preserve documented decision rules.
- **Requires experiment rerun:** No
- **Validation:** Recomputation from raw logs
- **Final status:** **NEEDS AUTHOR DECISION**

## STA-04 — Statistical-analysis inconsistency — severity: high

- **Affected files:** ResearchPaper_Revised/IEEE_Access_Result_Revised.docx table in 4.1
- **Experiment / optimizer / seed:** E1 table
- **Evidence:** Manuscript E1 values (Adam-NV 0.9337+/-0.0020 n=4; Adam 0.9319 n=2; AdamW 0.9314 n=2; AdaBelief 0.9267 n=2) are not reproduced by any seed subset of the logs. Seeds 0-3 last-5 means: Adam-NV 0.9327, Adam 0.9315, AdamW 0.9314, AdaBelief 0.9248. n=2 claims contradict logs for seeds 0-3 (4 per optimizer).
- **Corrective action:** Recomputed tables in analysis/results_recomputed/e1_*; comparison in reports/e1_manuscript_vs_recomputed.csv. Manuscript not edited.
- **Requires experiment rerun:** No
- **Validation:** analyze_e1.py
- **Final status:** **OPEN (manuscript edit required by authors)**

## STA-05 — Statistical-analysis inconsistency — severity: medium

- **Affected files:** e1_convergence/e1_results_table.csv
- **Experiment / optimizer / seed:** E1
- **Evidence:** SD column uses population SD (ddof=0) while all analysis code uses ddof=1; verified by recomputation (Adam-NV 0.0016 vs 0.0018). Generator script is not in the repo.
- **Corrective action:** analyze_e1.py reports ddof=1 explicitly. Original table preserved.
- **Requires experiment rerun:** No
- **Validation:** per-seed recomputation
- **Final status:** **RESOLVED (new table) / original retained**

## STA-06 — Statistical-analysis inconsistency — severity: medium

- **Affected files:** analysis/analyze_e1.py
- **Experiment / optimizer / seed:** E1 multiplicity
- **Evidence:** No runbook defines the E1 Holm family or an equivalence margin ('paired and equivalence tests' are promised but absent).
- **Corrective action:** Proposed families marked family_status=proposed_pending_author_confirmation in outputs; no equivalence margin invented, no verdict issued.
- **Requires experiment rerun:** No
- **Validation:** output columns
- **Final status:** **NEEDS AUTHOR DECISION**

## STA-07 — Statistical-analysis inconsistency — severity: low

- **Affected files:** analysis/analyze_e13_e15.py
- **Experiment / optimizer / seed:** E13/E14 constant arm
- **Evidence:** Constant-schedule default arm takes seeds 0-1 from part_b/ and seeds 2-3 from Ablation gamma=1.0 reruns (documented in a code comment as equivalent). Equivalence of the two sources is asserted, not tested here.
- **Corrective action:** Noted only.
- **Requires experiment rerun:** Unknown
- **Validation:** -
- **Final status:** **UNVERIFIED**

## STA-08 — Statistical-analysis inconsistency — severity: medium

- **Affected files:** adam-nv/analysis/analyze_exp07.py
- **Experiment / optimizer / seed:** EXP07/EXP08 label-noise analysis
- **Evidence:** The script printed a 'pre-registered decision' even when an arm lacked seeds: omitting the log_exp07_C folder (Adam-NV seeds 4-7) silently analysed Adam-NV with n=4 and gave M = +3.40 pp instead of the 8-seed +3.91 pp. This happened during the repair itself. The same gap was found in analyze_exp12.py (constant-0.95 arm at n=4 produced a 'PARTIAL / INCONCLUSIVE' outcome with no warning) and exists in analyze_exp08/09/11 by construction.
- **Corrective action:** Added one shared check_seed_balance() (analyze_exp07.py) with a loud unbalanced-arm WARNING and --strict-seeds (exit 2), called from analyze_exp07, 08, 09, 11 and 12. validate_all runs them with --strict-seeds.
- **Requires experiment rerun:** No
- **Validation:** tests: Exp07SeedBalanceTest, SeedGuardAcrossExperimentScriptsTests; validate_all steps 13, 13b, 13c, 13e, 13f, 13g (negative control)
- **Final status:** **RESOLVED**

## OBS-01 — Obsolete or broken script — severity: medium

- **Affected files:** README.md; EXP01_RUNBOOK.md; EXP03_RUNBOOK.md; e2_noisy_gradient/src/analyze_e2.py
- **Experiment / optimizer / seed:** E1/E2
- **Evidence:** analyze_e1.py and analyze_e1_protocol_v2.py are referenced but missing; analyze_e2.py is a 0-byte file; README E1 commands cd into e1_convergence/src, which does not exist at top level.
- **Corrective action:** Wrote analysis/analyze_e1.py (recomputation, NOT a reproduction of the lost script); fixed README paths; empty analyze_e2.py left as found and documented.
- **Requires experiment rerun:** No
- **Validation:** test_file_paths_named_in_docs_exist
- **Final status:** **RESOLVED (paths/scripts) ; lost scripts' numbers not reproducible**

## OBS-02 — Obsolete or broken script — severity: medium

- **Affected files:** adam-nv/EXP03_RUNBOOK.md; adam-nv/e1_convergence/exp03_*.zip, exp05_*.zip
- **Experiment / optimizer / seed:** EXP03, EXP05
- **Evidence:** PROTOCOL_V2.md (the pre-registered hypotheses and analysis plan) and analyze_exp03/05 are absent from the repository; EXP03/05 logs exist only as zips.
- **Corrective action:** Documented; EXP06 claim (+0.173 pt, p=0.031, 4/4) re-verified from exp05 zips with analyze_exp06.py (final-epoch metric).
- **Requires experiment rerun:** Not applicable
- **Validation:** analyze_exp06.py run on extracted logs
- **Final status:** **UNRESOLVED (documents not recoverable)**

## OPT-01 — Source-code or configuration mismatch — severity: medium

- **Affected files:** adam-nv/e2_noisy_gradient/src/adam_nv.py (docstring); e2_noisy_gradient/src/adam_nv.py
- **Experiment / optimizer / seed:** All Adam-NV runs
- **Evidence:** Docstring: gamma=0 'reduces exactly to Adam'. Measured (float64, 20 params): max |diff| to torch.optim.Adam is 4e-16 for <=150 steps, 1.4e-9 at 160, 6.3e-8 at 500, 2.8e-8 at 3000. Cause: P clamped at 1e-7 (0.9^t < 1e-7 after ~153 steps). The underflow RuntimeWarning also fires from step 132 even for Adam-equivalent settings (observed).
- **Corrective action:** NOT changed (file is hash-pinned; changing it would invalidate provenance of EXP07/08). Behaviour pinned by tests; claim documented as approximate beyond ~150 steps. Algorithm unchanged; no experiment is affected materially (<=1e-7 relative).
- **Requires experiment rerun:** No
- **Validation:** tests test_gamma_zero_equals_torch_adam_within_clamp_horizon, test_gamma_zero_long_horizon_deviation_is_small_but_documented
- **Final status:** **DOCUMENTED (docstring overstated)**

## OPT-02 — Unverified issue requiring further evidence — severity: low

- **Affected files:** tests/test_repair_optimizer.py
- **Experiment / optimizer / seed:** Optimizer verification limits
- **Evidence:** The independent NumPy reference was written from the code and docstring formulation by the repair author, not extracted from the paper's equations; Formulation/adam-NV_V1.py was not diffed against the paper.
- **Corrective action:** Stated as a limitation.
- **Requires experiment rerun:** No
- **Validation:** -
- **Final status:** **UNVERIFIED**

## DOC-01 — Documentation inconsistency — severity: medium

- **Affected files:** README.md; tests/; adam-nv/tests/
- **Experiment / optimizer / seed:** Test discoverability
- **Evidence:** Documented test command runs only tests/; nested adam-nv/tests (GONO, EXP06/07) must run with adam-nv/ as cwd and were never mentioned.
- **Corrective action:** tools/run_all_tests.py runs both roots; README updated.
- **Requires experiment rerun:** No
- **Validation:** tools/run_all_tests.py
- **Final status:** **RESOLVED**

## DOC-02 — Documentation inconsistency — severity: low

- **Affected files:** requirements.txt; check_logs.py defaults
- **Experiment / optimizer / seed:** Environment
- **Evidence:** Dependencies are ranges; experiments ran on torch 2.10.0+cu128 / torchvision 0.25.0+cu128 (check_logs.py defaults) but no lock or environment record exists.
- **Corrective action:** environment/ENVIRONMENT.md records declared, experiment and validation environments; requirements.txt not tightened (would not change past results).
- **Requires experiment rerun:** No
- **Validation:** -
- **Final status:** **MITIGATED**

## DAT-01 — Missing, truncated, duplicated, or incomplete logs — severity: high

- **Affected files:** adam-nv/e1_convergence/exp10_logs_{A,B}.zip; (absent) exp10_logs_C.zip; (absent) Exp.6 logs
- **Experiment / optimizer / seed:** EXP10: lr 3e-3 10-seed extension (Adam, Adam-NV, Adam b0.83, Adam b0.80)
- **Evidence:** UPDATE (push c534451): exp10_logs_C.zip (Adam and Adam-NV lr 3e-3, seeds 4-9; 24 runs) is now in the repository; all hashes verify and all runs are complete. With it, analyze_exp10.py gives Adam-NV - Adam over ten seeds = +0.172 pp, CI [+0.049, +0.294], raw p 0.0114, Holm 0.0572, higher in 9/10 (final-epoch: +0.159, Holm 0.128); seeds 4-9 alone: +0.155 [-0.081, +0.390], 5/6. STILL ABSENT: the Exp.6 logs (adam_b083 / adam_b080, seeds 0-3). Constants therefore have only 6 seeds (4-9): -0.020 and -0.026 pp vs Adam; the pre-registered decision is reported by the script as INCONCLUSIVE and 'provisional' (fewer than 10 paired seeds).
- **Corrective action:** Nothing invented. The ten-seed constant-beta1 contrasts and the 'Adam-NV minus constants' contrasts cannot be recomputed until the Exp.6 logs are pushed.
- **Requires experiment rerun:** Not by the repair: logs exist elsewhere or must be produced (GPU)
- **Validation:** validate_all step 13d (script declines)
- **Final status:** **PARTIALLY RESOLVED (Exp.6 control logs still missing)**

## DAT-02 — Documentation inconsistency — severity: low

- **Affected files:** adam-nv/e1_convergence/src/train_resnet_ex.py
- **Experiment / optimizer / seed:** EXP05 provenance
- **Evidence:** Push ba29996 deleted the file whose bytes (8aef0072) verify the EXP05 hashes.
- **Corrective action:** Bytes preserved (provenance/source_archive, newer adam-nv.zip, git history). The repaired tree keeps the file in place.
- **Requires experiment rerun:** No
- **Validation:** provenance verify: 48 VERIFIED_EXACT_RENAMED unchanged
- **Final status:** **RESOLVED (preserved)**

## DAT-03 — Missing, truncated, duplicated, or incomplete logs — severity: low

- **Affected files:** adam-nv/e2_noisy_gradient/src/{adam_nv_shuffle.py,part_b_resnet_v8_shuffle.py,check_exp11_logs.py,run_exp11_beta1_shuffle.sh}; adam-nv/analysis/analyze_exp11.py; adam-nv/EXP11_RUNBOOK.md
- **Experiment / optimizer / seed:** EXP11 beta1 shuffle control
- **Evidence:** UPDATE (push c534451): EXP11 logs are present: 36 runs (shuffle/step, shuffle/fixed, shuffle/none; 8 seeds each, 12 per zip), all complete. Their folder-level hashes name adam_nv_shuffle.py (8e4df34c) and part_b_resnet_v8_shuffle.py (b8612dff); the repository's working-tree files hash to exactly those values, so both were archived and now verify. analyze_exp11.py reproduces the reference M = +3.909 pp; its own outcome line is 'INCONCLUSIVE' (shuffle/step F = +2.40, D = +1.51 [+0.93,+2.08]; shuffle/fixed F = +2.81, D = +1.10 [+0.39,+1.81]). The 7 EXP11 unit tests pass.
- **Corrective action:** Archived adam_nv_shuffle.py@8e4df34c and part_b_resnet_v8_shuffle.py@b8612dff; registry rebuilt.
- **Requires experiment rerun:** Yes (GPU)
- **Validation:** provenance verify (EXP11 folders covered, claims exact); check_runs zip scan: 36 OK; validate_all step 13e
- **Final status:** **RESOLVED (by author-supplied EXP11 logs)**

## DAT-04 — Missing, truncated, duplicated, or incomplete logs — severity: low

- **Affected files:** adam-nv/e2_noisy_gradient/logs/exp12_logs_{1,2,3,4}.zip; adam-nv/analysis/analyze_exp12.py
- **Experiment / optimizer / seed:** EXP12: constant beta1 0.92 / 0.95 under 20% label noise (16 runs, seeds 0-7)
- **Evidence:** All 16 runs complete (60 epochs, metadata present, deterministic); the five pinned folder hashes verify. The 'adam-nv.zip' snapshot also holds nested copies of each zip (every run appears twice in scans). analyze_exp12.py: constant 0.92 - Adam = +0.44 pp [-0.82,+1.71]; constant 0.95 - Adam = -1.03 [-2.21,+0.16]; smooth - const 0.92 = -1.91 [-3.52,-0.30] (Holm 0.10); smooth - const 0.95 = -0.45 [-1.32,+0.43]; script outcome: PARTIAL / INCONCLUSIVE. File names carry no beta1, so analyses must pass each call's folder explicitly.
- **Corrective action:** None needed. Results recorded, not interpreted.
- **Requires experiment rerun:** No
- **Validation:** validate_all steps 13f (balanced, --strict-seeds) and 13g (unbalanced must fail)
- **Final status:** **RESOLVED (data verified)**

## DAT-05 — Unverified issue requiring further evidence — severity: low

- **Affected files:** adam-nv/benchmarks/timing_v2_results.zip; adam-nv/benchmarks/benchmark_optimizer_overhead_v2.py; adam-nv/TIMING_RUNBOOK.md
- **Experiment / optimizer / seed:** Optimizer overhead (one Tesla T4 session, 20 rounds)
- **Evidence:** The benchmark script's hash (fb71196a) matches the working-tree file and is now archived. Summary values reproduce the draft's text: total step 126.5 vs 108.0 ms (+17.1%, CI [16.9,17.5]%), optimizer step 27.3 vs 3.8 ms (7.2x), state 170.5 vs 85.3 MiB. Timings depend on the hardware/session and cannot be reproduced here (no GPU).
- **Corrective action:** Archived the benchmark source; claims checked against the stored summary only.
- **Requires experiment rerun:** Yes (GPU) to reproduce
- **Validation:** reports/paper_draft_claim_check.csv
- **Final status:** **UNVERIFIED (hardware-specific; summary consistent)**

## DOC-03 — Documentation inconsistency — severity: medium

- **Affected files:** adam-nv/paperdraft/adamnv_paper_draft.pdf (newer, in repo); adam-nv/paperdraft/adamnv_paper_draft_uploaded_20261026T0658Z.pdf (older, supplied separately)
- **Experiment / optimizer / seed:** Paper draft (all sections with numbers)
- **Evidence:** Recomputed from raw logs by reports/_build_paper_check.py over the claims of both versions: 95 MATCH at printed precision; 6 NOT VERIFIABLE; 1 DIFFERS. DIFFERS: Table I lists the overhead benchmark as '10 repeats' while the newer draft's own Section VII and the stored v2 results say 20 rounds (the older draft's '10 repeats' fits its own v1 benchmark, whose results are not in the repository). NOT VERIFIABLE: AdaBelief over 10 seeds (repository has 4; their values: mean 92.48, -0.66 pp [-0.83,-0.50] vs Adam, epochs-to-90% / AUC); constant-beta1 0.83/0.80 ten-seed contrasts and Adam-NV-minus-constants (Exp.6 logs missing); and, for the older draft only, the v1 timing numbers (31.0 vs 4.2 ms; 127.4 vs 105.5 ms; +20.8%). VERIFIED, including items first thought unverifiable: Table VII element-level beta1 distribution (reproduces exactly from the EXP11 'none' runs' beta1_dist_*.jsonl; that arm is also bit-identical to the original Adam-NV label-noise runs), Table II epochs-to-90% and validation-loss AUC, the GONO exact-correction variant, and the logged beta1 statistics. The drafts already use the CORRECTED E14/E15 values, not the stale committed CSV, and correctly state that a GONO log lost its final row and the repeat matched the other 59 epochs. The older draft's TODO to reconcile source files with the logged hashes can only be met partially (see PRV-01/02/04: 48 runs unreproducible, 96 with no hash).
- **Corrective action:** Report only (reports/PAPER_DRAFT_CHECK.md, paper_draft_claim_check.csv, UPLOADED_DRAFT_COMPARISON.md). Neither draft was edited.
- **Requires experiment rerun:** No
- **Validation:** python reports/_build_paper_check.py
- **Final status:** **OPEN (author edit of Table I; 6 claims need missing logs)**
