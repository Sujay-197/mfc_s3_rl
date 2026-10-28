# Results — PPO on Super Mario Bros. 1-3

Evaluation: 20 episodes per run with the **stochastic** policy, plus one **deterministic (greedy)** pass.

The environment is deterministic, so the greedy policy produces an identical trajectory every run; it is reported as a single pass/fail. Clear rate below is the stochastic figure, which is the meaningful measure of robustness.
The flagpole is at x ≈ 1265 (measured from the runs).

| run | clear rate | actions | seed | steps trained | first clear (steps) | mean reward | mean x | best x | greedy clears |
|---|---|---|---|---|---|---|---|---|---|
| lvl13_simple_seed0 | **0%** | simple | 0 | 2,089,837 | — | 69.4 | 758 | 783 | no |

**Best run: `lvl13_simple_seed0` — clears World 1-3 in 0% of 20 episodes.**

### Note on the deterministic policy

The environment is deterministic, so the greedy pass replays a single trajectory: it reads as one pass/fail, not a rate.
At the final checkpoint the greedy policy fails to finish for `lvl13_simple_seed0` — the same narrow failure of action selection observed on 1-1, where sampling from the identical policy still clears the level.
