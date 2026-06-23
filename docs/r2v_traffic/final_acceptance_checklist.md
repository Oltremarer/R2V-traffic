# Final Acceptance Checklist

- [x] Generated smoke/main/ablation commands parse against the target runner.
- [x] Runner dry-runs baseline-off and R2V-on paths.
- [x] `gate_variant` changes runtime admission behavior.
- [x] Admission mode is explicit: default `weights_only`, opt-in `weights_plus_repaired` fails closed for admitted rows without repaired payloads and marks rejected repair proposals separately.
- [x] External score-artifact repair payloads must include their own `transition_id` and `sample_id`.
- [x] `not_rare_to_val + full` interface exists and can build ordinary transition buffers with explicit proxy repair metadata.
- [x] `not_val_to_val` interface exists and is tested.
- [x] Gate ablation interfaces exist and are tested.
- [x] Sampling ablation command interfaces exist and are tested.
- [x] Runner `r2v_admitted_weight` uses exact admitted sample weight semantics.
- [x] Weighted artifact validation fails closed.
- [x] Weighted artifact row roles are checked so repaired proposals cannot masquerade as admitted samples.
- [x] Runtime weighted outputs include v2 traffic artifact schema fields.
- [x] Runtime R2V pair sampling rejects legacy v1-only weighted artifacts.
- [x] Paper artifact manifest validates weighted-transition artifacts against the v2 traffic schema.
- [x] Paper artifact manifest requires diffusion weighted-transition artifacts to match bundled diffusion-score artifacts.
- [x] Paper artifact manifest rejects readiness reports that are not `READY`.
- [x] Paper artifact manifest validates aggregation artifacts against the result aggregation schema and required five metrics.
- [x] Paper artifact manifest requires aggregation artifacts to include performance/integrity input hashes and match bundled artifacts.
- [x] Paper artifact manifest validates performance artifacts contain all five required traffic metrics.
- [x] Paper artifact manifest validates integrity artifacts contain R2V candidate/admission/gate summaries.
- [x] Missing smoke/main prerequisites are reported by an executable readiness checker.
- [x] Strict paper readiness blocks proxy diffusion artifacts and proxy repair metadata.
- [x] Proxy candidate runs without a real score artifact are labeled as proxy backends, not diffusion.
- [x] Main-pipeline R2V commands pass each seed's real diffusion artifact through `--r2v_artifact_path`.
- [x] Rarity is not a final admission gate; value/OOD, support, and dynamics/safety decide admission.
- [x] V2 traffic metadata IDs must match row transition/sample IDs.
- [x] Baseline path keeps R2V off.
- [x] Performance/status aggregation is separated.
- [x] Documentation directory is complete.
- [ ] Full CityFlow smoke was run on the target experiment machine. Local attempt was blocked by missing `data/Jinan/3_4/anon_3_4_jinan_real.json`.
- [ ] Main 3-seed traffic metrics were generated.
- [ ] Paper performance table was aggregated from completed evaluation rows.

## Acceptance status

Engineering/interface migration is accepted. Paper-result acceptance waits for real experiment execution.
