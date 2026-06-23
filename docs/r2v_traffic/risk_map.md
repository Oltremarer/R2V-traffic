# Risk Map

| risk_id | risk | detection | mitigation |
|---|---|---|---|
| R01 | Rare is mistaken for valuable | rare-only performs like full without value/support evidence | Keep rare and value gates separate; include rare-only ablation. |
| R02 | Zero reward is treated as corrupted | preprocessing drops zero-reward rows | Never filter solely by zero reward; inspect horizon outcomes. |
| R03 | Generator becomes value oracle | paper text ranks by diffusion score | State that diffusion is candidate discovery or proposal only. |
| R04 | Admission uses off-support traffic states | support/dynamics gates fail or are bypassed | Main path uses full gate; ablations are marked as ablations. |
| R05 | Bellman target is changed accidentally | learner code changes target computation | Keep R2V changes at replay measure, pair sampling, or admitted set. |
| R06 | Baseline behavior changes when R2V is off | tests show `--r2v_sampling_mode off` still uses weights | Fail closed if weighted artifacts are passed while sampling is off. |
| R07 | Status is confused with performance | status.json appears in result ranking | Aggregator separates performance metrics from integrity/status. |
| R08 | Weak baselines inflate gains | only random/fixed-time included | Add MaxPressure and repo-native learned baselines before paper claims. |
| R09 | True diffusion repair is overclaimed | code only uses conservative existing-transition weighting | Document current boundary: diffusion backend interface is represented, but full state repair needs future integration unless implemented later. |
| R10 | Proxy smoke artifact is mislabeled as diffusion evidence | weighted rows without a score artifact show `r2v_generative_backend=diffusion` | Candidate builder records proxy candidate model unless a real score artifact path is supplied; strict manifests block proxy diffusion claims. |
| R11 | ER baseline drift is mistaken for R2V gain | `baseline_recent` is compared to `r2v_uniform` or undocumented ER modes | Pair each ER baseline with the matching R2V overlay and report `er_baseline_mode` plus `er_r2v_combine`. |
| R12 | Final pipeline preflights diffusion artifacts but runner uses proxy scoring | main R2V command lacks `--r2v_artifact_path` | Main-pipeline validation checks per-seed diffusion artifact wiring; manifest checks weighted score-artifact provenance. |
| R13 | Proxy safety gate is overclaimed as strict dynamics proof | paper says computed gate proves next-state consistency | Reserve strong dynamics/recovery-distance claims for external repair artifacts with final gate metadata or a dedicated dynamics checker. |

## Conclusion

The safest paper framing is conservative: R2V-Traffic changes the empirical replay distribution through proposed and admitted traffic transitions. It does not claim that rare, low reward, or generated transitions are automatically good.
