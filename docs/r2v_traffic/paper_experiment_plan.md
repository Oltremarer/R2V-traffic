# Paper Experiment Plan

## Main question

Does R2V-Traffic improve traffic signal control metrics over the same baseline replay path when using `diffusion + not_rare_to_val + full` admission?

## Smoke

Run one small Jinan/CityFlow setting for baseline and R2V with one seed. Verify that artifacts are valid, logs are present, and traffic metrics are generated.

## Main

Run baseline vs `R2V-diffusion-not_rare_to_val-full` over seeds 0, 1, and 2 on a formal map. Report average travel time, queue length, delay, throughput, and reward.

## Ablations

Run gate ablations (`full`, `no_support`, `no_ood`, `no_dynamics`), story ablations (`not_rare_to_val`, `not_val_to_val`), and sampling ablations (`admitted_only`, `random_same_count`, `shuffled_value`, `inverted_rarity`).

## Evidence split

Candidate counts, admitted counts, gate pass rates, hashes, and status files are integrity evidence. Only completed traffic metric rows support performance claims.
