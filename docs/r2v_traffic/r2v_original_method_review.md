# Original R2V Method Review

## Source snapshot

- Original repo: `/Users/azure/Documents/R2V`
- Branch: `main`
- Commit: `77a97317f9f2b4f11c80f6c7a90fa770a0cc93ee`
- Remote comparison: `0 ahead / 0 behind origin/main`
- Dirty state during review: clean

## Pipeline

Original R2V flattens an offline RL dataset into a dense transition matrix `x = [s, a, r, s']`, while terminal flags are preserved as separate dataset fields. The method then:

1. scores transitions with a detector;
2. selects rare or reference candidate sets;
3. trains or applies a diffusion/flow repair model;
4. evaluates support, OOD/value, and dynamics gates;
5. admits accepted transitions into an enhanced replay artifact;
6. passes sample weights to the downstream RL runner.

## Detector

The detector input is a transition matrix. The output is a score per transition plus candidate/reference splits. The detector can use proxy reconstruction/norm scores or diffusion/flow scoring adapters.

## Repair stories

`not_rare_to_val` starts from non-rare transitions and tries to propose valuable admitted replay mass. `not_val_to_val` starts from rare or low-value candidates and tries to repair/propose them toward admitted valuable patterns. In both cases, repair is proposal, not final judgment.

## Gates

Support gate checks behavior-support plausibility. OOD/value gate checks whether the proposed transition has useful value/progress rather than mere rarity. Dynamics gate checks transition consistency. Admission is the conjunction or configured ablation of these gates.

## Bellman target boundary

Original R2V changes the empirical replay measure, sample weights, or admitted dataset. It does not require changing the RL Bellman target. This boundary must be preserved in traffic.
