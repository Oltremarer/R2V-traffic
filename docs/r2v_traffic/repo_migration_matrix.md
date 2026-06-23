# R2V to R2V-Traffic Migration Matrix

| R2V source | source role | traffic target | target role | reuse | adapter | rewrite | risk | test |
|---|---|---|---|---|---|---|---|---|
| `synther/adg/schema.py`, `data.py` | D4RL transition matrix | `pareto/data/schema.py::TransitionRecord` | traffic JSONL transition | no | yes | no | schema mismatch | schema and validation tests |
| `synther/adg/detector.py` | rare score and split | `pareto/r2v/traffic_candidate_selector.py` | rarity and candidate scoring | conceptual | yes | partial | rare != value confusion | candidate selector tests |
| `diffusion_adapter.py`, `flow_adapter.py` | generative score/denoise | `pareto/r2v/generative_scorer.py`, traffic config | score artifact and backend hook | conceptual | yes | partial | true traffic repair not complete | schema and command-plan tests |
| `recovery.py` | repair projection | traffic repair story metadata | conservative proposal status | no | yes | yes for real repaired states | illegal traffic states | gate/story tests |
| `gates.py` | support/OOD/dynamics gates | traffic rare/value/support/safety then v2 rare/ood/support/dynamics | admission masks | conceptual | yes | partial | semantics differ | gate ablation tests |
| `admission.py` | build enhanced dataset/weights | `apply_candidate_weights` and pair builder | weighted transition metadata | conceptual | yes | partial | append-vs-weight mismatch | artifact validation and pair tests |
| replay artifact code | load/save enhanced data | `artifact_validation.py`, `traffic_artifact_schema.py` | fail-closed JSONL validation | no | yes | no | schema drift | new v2 artifact tests |
| offline runner scripts | build R2V train/eval plans | `jinan_pair_ablation_runner.py`, `traffic_experiment_plan.py` | dry-run and experiment plan | no | yes | no | CLI mismatch | command-plan tests |
| original tests | R2V invariants | `tests/pareto/test_r2v_*` | traffic invariants | no | no | yes | fixture drift | targeted pytest |

## Conclusion

Traffic should not import the original D4RL matrix implementation blindly. The stable migration unit is the method contract: candidate scoring, proposal, independent gates, admission weights, and downstream replay sampling.
