# Traffic Gate Design

## Support gate

Support gate checks whether a candidate stays close to behavior support or plausible traffic state support. It protects against impossible queue/phase combinations and far-off-distribution proposals.

## OOD / value gate

The current v1 selector uses a legacy gate name `value`; the v2 artifact schema maps that legacy signal to `ood` for the traffic method vocabulary. This is an alias, not permission to collapse value and OOD. In paper text, describe it as the OOD/value gate and report diagnostics separately when possible.

This gate checks that a candidate carries progress or utility and is not merely rare.

## Dynamics / consistency gate

The legacy v1 selector calls this `safety`; the v2 schema maps it to `dynamics`. It protects traffic consistency: queue evolution, phase-action legality, reward consistency, and next-state plausibility.

In the proxy/computed path, this is still a conservative safety-style signal, not proof that a learned traffic dynamics model has certified next-state consistency. Strong paper wording should reserve "dynamics consistency" claims for real external repair artifacts that provide final gate metadata or a dedicated dynamics checker.

## Gate variants

The implementation supports:

- `full`: support + OOD/value + dynamics;
- `no_support`;
- `no_ood`;
- `no_dynamics`.

The full gate is the main method. Ablations are diagnostic; they should not be used as the safest default.

Rarity is deliberately not listed as a final admission gate. It remains a detector/source-set signal and an ablation dimension, but final admission is decided by value/OOD, support, and dynamics/safety. This prevents "rare" from being silently treated as "valuable".

## Fail-closed rule

If an artifact is missing required gate masks, IDs, or finite positive weights, validation rejects it. Bad data should not silently enter training.
