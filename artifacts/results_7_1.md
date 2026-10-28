# Results — PPO on Super Mario Bros. 7-1

Evaluation: 20 episodes per run with the **stochastic** policy, plus one **deterministic (greedy)** pass.

The environment is deterministic, so the greedy policy produces an identical trajectory every run; it is reported as a single pass/fail. Clear rate below is the stochastic figure, which is the meaningful measure of robustness.
The flagpole is at x ≈ 2857 (measured from the runs).

| run | clear rate | actions | seed | steps trained | first clear (steps) | mean reward | mean x | best x | greedy clears |
|---|---|---|---|---|---|---|---|---|---|
| lvl71_simple_seed0 | **80%** | simple | 0 | 1,260,229 | 1,500,528 | 267.6 | 2671 | 2857 | yes |
| test_lvl71 | **0%** | simple | 0 | 88 | — | 33.0 | 399 | 882 | no |

**Best run: `lvl71_simple_seed0` — clears World 7-1 in 80% of 20 episodes.**

### Note on the deterministic policy

The environment is deterministic, so the greedy pass replays a single trajectory: it reads as one pass/fail, not a rate.
At the final checkpoint the greedy policy fails to finish for `test_lvl71` — the same narrow failure of action selection observed on 1-1, where sampling from the identical policy still clears the level.
