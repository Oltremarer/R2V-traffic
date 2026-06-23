# R2V-Traffic Literature Survey

## How this was completed

I checked recent and classic traffic signal control, replay, offline RL, offline-to-online RL, and generative augmentation work using web search plus local code inspection. The goal was not to prove R2V-Traffic works before experiments; it was to position the method with evidence boundaries that are safe for a paper.

## Core findings

Experience replay in traffic signal control is usually used as a practical stabilizer for online deep RL or as an offline dataset for controller learning. In legacy LLMLight-style code, replay samples are state, action, next state, reward windows, and time metadata. In this repo's Pareto stack, transitions are schema-first JSON rows with `transition_id`, `sample_id`, `obs_features`, `next_obs_features`, `action`, `env_reward`, objectives, `done`, and metadata.

Traffic replay buffers usually store local or network intersection states: phase, lane vehicle counts, queue/waiting features, pressure-like features, action/phase, rewards, next state, and rollout identifiers. Sampling is often uniform in baseline implementations. Prioritized replay exists in general RL, and some offline/traffic works use return or importance weighting, but priority is not rarity.

Recent TSC work increasingly focuses on data efficiency, transfer, offline learning, safety, and robustness. Classic baselines remain important: MaxPressure, PressLight, CoLight, FRAP, fixed-time, and sometimes MPLight/Advanced-CoLight/LLMLight variants. CityFlow, Jinan/Hangzhou/New York style roadnets, and LibSignal/RESCO-style benchmark protocols are common.

Generative replay and trajectory augmentation, such as SynthER and diffusion trajectory augmentation, support the idea that generated or repaired transitions can be useful candidate data. They do not support using the generator as a final value oracle. For traffic, this is stricter because a rare congestion pattern may be dangerous, off-support, or simply unhelpful.

## R2V-Traffic gap

R2V-Traffic fills a replay-mining gap: traffic controllers can benefit from better empirical replay distributions, but rare traffic states are not automatically valuable. The method should therefore be framed as proposal plus admission:

1. detector proposes candidate transitions;
2. repair/proposal suggests a direction toward valuable patterns;
3. support, OOD/value, and dynamics gates decide admission;
4. downstream traffic RL keeps its learner and Bellman target unchanged.

## Evidence boundary

The literature supports studying replay selection and augmentation. The paper must still prove traffic performance with actual metrics: average travel time, queue length, delay, throughput, and reward. Artifact completion, candidate counts, gate pass rates, and hashes are integrity/status evidence, not performance evidence.

Useful source links include [PressLight](https://doi.org/10.1145/3292500.3330949), [CoLight](https://doi.org/10.1145/3357384.3357902), [FRAP](https://arxiv.org/abs/1905.04722), [CityFlow](https://arxiv.org/abs/1905.05217), [LibSignal](https://arxiv.org/abs/2211.10649), [CrossLight](https://doi.org/10.1145/3637528.3671927), [OffLight](https://arxiv.org/abs/2411.06601), [PLight/PRLight](https://arxiv.org/html/2503.08728v1), [PER](https://arxiv.org/abs/1511.05952), [HER](https://arxiv.org/abs/1707.01495), [RLPD](https://arxiv.org/abs/2302.02948), [SynthER](https://openreview.net/forum?id=6jNQ1AY1Uf), and [GTA](https://openreview.net/forum?id=kZpNDbZrzy).

## Replay mechanism matrix

This matrix is the paper-facing way to avoid overgeneralizing from broad replay claims.

| paper/work | setting | replay or data role | tuple fields | sampling/reweighting | generator or augmentation | admission/filter | maps/metrics relevance |
|---|---|---|---|---|---|---|---|
| PressLight | online TSC baseline | DQN-style replay stabilizes learned max-pressure control | intersection state, phase/action, reward, next state | mostly uniform implementation-style replay | none | learned Q update, no R2V-style candidate gate | CityFlow-style TSC; pressure reward, travel/queue metrics |
| CoLight | online MARL TSC | replay supports graph-attention multi-intersection learning | local observations plus neighbor/attention context | baseline replay, not rarity-driven | none | no generator/admission split | multi-intersection benchmark baseline |
| FRAP | online TSC generalization | replay is standard deep RL infrastructure | phase competition state/action/reward/next state | not a rare-event miner | none | value learning only | useful learned baseline family |
| CityFlow / LibSignal / TSC benchmarks | simulator/benchmark | defines reproducible environment and metrics rather than a replay algorithm | roadnet, traffic file, signal phases, per-vehicle timing | benchmark protocol dependent | none | evaluation protocol | map/metric evidence for Jinan/Hangzhou/New York, travel time, queue, delay, throughput |
| Offline RL for Road Traffic Control / DataLight / D2TSC | offline TSC | fixed datasets replace unsafe online exploration | historical/simulated state, action, reward, next state, scenario metadata | conservative or model-based offline use, not rare=valuable | sometimes model-based or reward inference | support/pessimism matters | supports offline data reuse motivation |
| DTLight / CrossLight | offline-to-online TSC | offline pretraining plus online adaptation | cross-city or trajectory datasets | transfer/adaptation, sometimes data mixing | representation or transfer mechanisms | no R2V gate stack | supports data-efficient adaptation framing |
| OffLight | offline MARL TSC | heterogeneous offline behavior data | multi-agent transition records with behavior-source concerns | return/importance-style sampling, behavior-aware reuse | no diffusion repair | sampling is not final value proof | nearest traffic replay-selection baseline |
| PER | general RL replay | priority changes empirical sample distribution | transition plus priority/error | TD-error priority, importance correction | none | priority is not rarity and not traffic admission | indirect baseline for reweighting |
| RLPD | general offline-to-online RL | mixes offline data with online replay | offline and online transitions | replay mixing policy | none | no traffic dynamics gate | indirect replay-mixing baseline |
| SynthER / GTA | general RL generative replay | generated samples augment replay | transition or trajectory-like samples | generated candidates enter learner after method-specific checks | diffusion/generative models | not traffic-specific admission proof | indirect evidence for candidate generation only |

## Nearest-baseline matrix for R2V-Traffic

| nearest baseline | what it already covers | what R2V-Traffic adds | claim boundary |
|---|---|---|---|
| Uniform ER | stable minibatch reuse | non-uniform empirical replay measure from admitted candidates | must beat uniform on actual traffic metrics |
| PER / TD-error priority | prioritizes prediction surprise | separates rarity, value, support, and dynamics gates | TD error or rarity alone is not value |
| Return or importance replay in offline TSC | behavior-aware or return-aware sample weighting | generator/proposal plus independent traffic admission | compare as replay-selection baseline, not as strawman |
| Offline-to-online replay mixing | uses historical data for safer adaptation | mines and weights candidate transitions inside the traffic replay distribution | does not remove need for online/simulator evaluation |
| Generative replay / SynthER / GTA | shows generated candidates can help RL | traffic-specific support/value/dynamics admission before replay use | generator is not an oracle |
| Local proxy weighted sampling | runnable smoke interface | true diffusion score/repair artifact path and strict paper readiness | proxy smoke is not paper diffusion evidence |

## Evidence hygiene

Claims in this project should be labeled by evidence source:

- literature evidence: papers and benchmarks listed in `paper_table.csv`;
- local-code evidence: current repo files, tests, and generated artifact schemas;
- internal policy: reporting rules such as keeping status artifacts out of performance tables.

Internal policy is important for engineering discipline, but it should not be cited as literature evidence. Paper claims about performance need traffic metrics, while claims about implementation integrity can cite schema checks, hashes, and readiness reports.

## Remaining risks

The strongest risk is overclaiming. R2V-Traffic can claim a principled proposal-admission replay framework before experiments, but it can only claim performance gains after finished traffic evaluations. Another risk is baseline weakness: a paper result should include strong traffic baselines, not only random or fixed-time control.
