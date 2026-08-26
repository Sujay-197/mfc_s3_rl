# Results — PPO on Super Mario Bros. 1-1

Evaluation: 20 episodes per run with the **stochastic** policy, plus one **deterministic (greedy)** pass.

The environment is deterministic, so the greedy policy produces an identical trajectory every run; it is reported as a single pass/fail. Clear rate below is the stochastic figure, which is the meaningful measure of robustness.
The flagpole is at x ≈ 3160.

| run | clear rate | actions | seed | steps trained | first clear (steps) | mean reward | mean x | best x | greedy clears |
|---|---|---|---|---|---|---|---|---|---|
| simple_seed0 | **75%** | simple | 0 | 1,013,214 | 44,532 | 285.0 | 2887 | 3161 | yes |
| right_only_seed2 | **75%** | right_only | 2 | 1,001,089 | 108,160 | 268.4 | 2725 | 3161 | yes |
| simple_seed1 | **75%** | simple | 1 | 1,007,038 | 71,196 | 257.8 | 2612 | 3161 | yes |

**Best run: `simple_seed0` — clears World 1-1 in 75% of 20 episodes.**

### Note on the deterministic policy

Earlier in training (~500K steps) the greedy (argmax) policy stalled: it ran to x~2370 and then stood still until the in-game clock expired — roughly 90% of the episode motionless, with only ~10% of steps showing any movement — while sampling from that same policy cleared the level. By 1M steps the behaviour had resolved: every run's greedy policy now clears the level, so the stall was a transient mid-training artifact rather than a property of the converged agent.

Either way it indicates a narrow failure of action selection rather than of the learned representation, and it is consistent with the finding in [yumouwei/super-mario-bros-reinforcement-learning](https://github.com/yumouwei/super-mario-bros-reinforcement-learning) that a single-frame agent gets stuck and that temporal context helps the agent commit to jumps.
