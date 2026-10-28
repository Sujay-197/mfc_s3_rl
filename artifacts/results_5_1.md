# Results — PPO on Super Mario Bros. 5-1

Evaluation: 20 episodes per run with the **stochastic** policy, plus one **deterministic (greedy)** pass.

The environment is deterministic, so the greedy policy produces an identical trajectory every run; it is reported as a single pass/fail. Clear rate below is the stochastic figure, which is the meaningful measure of robustness.
The flagpole is at x ≈ 3177 (measured from the runs).

| run | clear rate | actions | seed | steps trained | first clear (steps) | mean reward | mean x | best x | greedy clears |
|---|---|---|---|---|---|---|---|---|---|
| lvl51_simple_seed0 | **75%** | simple | 0 | 2,256,236 | 354,316 | 265.0 | 2685 | 3177 | yes |

**Best run: `lvl51_simple_seed0` — clears World 5-1 in 75% of 20 episodes.**

### Note on the deterministic policy

The environment is deterministic, so the greedy pass replays a single trajectory: it reads as one pass/fail, not a rate.
At the final checkpoint every run's greedy policy finishes the level.
