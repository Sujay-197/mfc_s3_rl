# Results — PPO on Super Mario Bros. 1-2

Evaluation: 20 episodes per run with the **stochastic** policy, plus one **deterministic (greedy)** pass.

The environment is deterministic, so the greedy policy produces an identical trajectory every run; it is reported as a single pass/fail. Clear rate below is the stochastic figure, which is the meaningful measure of robustness.
The flagpole is at x ≈ 3161 (the level's recorded value).

| run | clear rate | actions | seed | steps trained | first clear (steps) | mean reward | mean x | best x | greedy clears |
|---|---|---|---|---|---|---|---|---|---|
| lvl12_complex_seed1 | **65%** | complex | 1 | 1,102,900 | 324,993 | 244.3 | 2573 | 3161 | no |
| lvl12_complex_seed0 | **25%** | complex | 0 | 1,084,091 | 398,682 | 205.7 | 2123 | 3161 | no |

**Best run: `lvl12_complex_seed1` — clears World 1-2 in 65% of 20 episodes.**

### Note on the deterministic policy

The environment is deterministic, so the greedy pass replays a single trajectory: it reads as one pass/fail, not a rate.
At the final checkpoint the greedy policy fails to finish for `lvl12_complex_seed1`, `lvl12_complex_seed0` — the same narrow failure of action selection observed on 1-1, where sampling from the identical policy still clears the level.
