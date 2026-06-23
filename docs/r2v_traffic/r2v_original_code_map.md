# Original R2V Code Map

| component | original R2V role | migration note |
|---|---|---|
| `synther/adg/schema.py` | dense transition layout | Traffic needs JSONL IDs and metadata instead of D4RL matrix only. |
| `synther/adg/data.py` | dataset to transition matrix conversion | Traffic adapter must flatten `TransitionRecord` fields. |
| `synther/adg/detector.py` | detector scores and rare/reference splits | Traffic already has proxy and score-artifact adapter. |
| `synther/adg/diffusion_adapter.py` | diffusion scoring and denoise hooks | Traffic currently exposes diffusion backend config but uses conservative artifact interface. |
| `synther/adg/recovery.py` | repair projection | Traffic repair must respect legal phases, queues, next-state consistency. |
| `synther/adg/gates.py` | support, OOD/value, dynamics masks | Traffic maps these to support, OOD/value, dynamics/safety. |
| `synther/adg/admission.py` | enhanced dataset and sample weights | Traffic uses weighted transition metadata and pair sampling. |
| `synther/replay_enhancement/artifact.py` | artifact writer/loader | Traffic uses JSONL validation and v2 wrapper schema. |
| `scripts/run_r2v_offline.py` | offline runner command plan | Traffic uses command planners and dry-run experiment specs. |
| `tests/test_adg_*.py` | method invariants | Traffic tests should preserve invariants, not fixture shapes. |

## Conclusion

Direct code reuse is limited because the schemas differ. Conceptual reuse is strong: detector, proposal, independent gates, admission, and replay weighting are the migration spine.
