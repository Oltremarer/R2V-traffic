# Paper Motivation

Traffic signal control relies heavily on replayed experience because online exploration is expensive and sometimes unsafe. Standard experience replay treats most transitions uniformly, and PER-like ideas prioritize learning error rather than traffic-specific value.

Rare traffic events matter because they may reveal congestion, coordination failures, or spillback-like pressure patterns. They are also dangerous to overuse: rare does not mean valuable, and a rare state can represent bad behavior, sensor noise, or off-support dynamics.

R2V-Traffic therefore decomposes the problem:

1. use a detector or generator to find candidate replay mass;
2. use repair/proposal to suggest useful directions;
3. let independent gates decide admission;
4. change replay distribution without changing the base traffic RL target.

The contribution is a proposal-admission replay framework for traffic signal control with explicit support, OOD/value, and dynamics gates plus ablations that test whether rarity, value, and gating are actually doing separate work.
