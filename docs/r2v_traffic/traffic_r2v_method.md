# Traffic R2V Method

## Method overview

R2V-Traffic ports the original R2V idea into traffic signal control by changing replay selection rather than the RL target.

1. Build traffic transitions with stable IDs.
2. Score candidate rarity with a proxy detector or diffusion score artifact.
3. Apply repair/proposal story:
   - `not_rare_to_val`: ordinary source transition moves toward admitted valuable pattern;
   - `not_val_to_val`: rare or low-value source candidate moves toward admitted valuable pattern.
4. Apply independent support, OOD/value, and dynamics gates.
5. Emit weighted transition artifacts and gate metadata.
6. Use weights in downstream pair/replay sampling when R2V is explicitly enabled.

## Admission modes

The conservative current path is weights-only: it reweights existing traffic transitions and metadata. A future full repaired-state path must additionally prove phase legality, queue evolution consistency, and reward consistency before appending repaired rows.

## Backbone boundary

Traffic controllers and downstream RL code remain responsible for their Bellman or PPO targets. R2V changes what data is sampled and how strongly it is weighted.

## Current implementation boundary

The code now includes v2 artifact validation, config/CLI flag schema, experiment command planning, and performance-vs-integrity aggregation. Existing candidate selector and pair builder provide the main opt-in replay weighting path.
