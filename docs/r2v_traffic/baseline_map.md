# Baseline Map

## How this was completed

I combined the literature survey with the local runner structure. The goal is to avoid a paper where R2V only beats weak baselines.

| baseline | role | why it matters |
|---|---|---|
| FixedTime / Random | sanity baseline | Useful for smoke, not enough for a paper claim. |
| MaxPressure | strong non-learning baseline | Theory-grounded traffic control; must be included where supported. |
| PressLight | strong RL baseline | Connects RL design to max-pressure reward/state choices. |
| CoLight / Advanced-CoLight | strong MARL baseline | Tests network coordination under learned control. |
| FRAP | phase-competition baseline | Useful generalization reference. |
| LLMLight / MPLight / existing repo runners | local target baselines | Keep original target behavior unchanged when R2V is off. |
| Offline TSC / offline-to-online methods | modern comparison family | Closest literature framing for replay reuse and adaptation. |
| R2V ablations | internal controls | full, no_support, no_ood, no_dynamics, admitted_only, random_same_count, shuffled_value, inverted_rarity. |

## Conclusion

For the first smoke, compare baseline vs `R2V-diffusion-not_rare_to_val-full`. For a paper table, group methods by environment, map, seed, and backbone. Do not mix MaxPressure heuristic metrics with offline pair-training integrity artifacts.
