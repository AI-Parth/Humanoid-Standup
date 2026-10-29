# HumanoidStandup-v5: PPO → SAC/TD3 → CrossQ → BAC

Single-file implementations of the methods that have published results on Gymnasium's
`HumanoidStandup-v5`, sharing one env setup and one set of "is it standing?" metrics.

| File | What it is |
|---|---|
| `common.py` | env factory (actions rescaled to [-1, 1]), running mean/std, posture metrics, logger, left/right mirror map |
| `ppo.py` | PPO (CleanRL-style), `--rpo-alpha 0.5` for RPO, `--sym-coef 0.02` for the symmetric-action loss |
| `offpolicy.py` | `--algo sac \| td3 \| crossq \| bac` |
| `evaluate.py` | runs a checkpoint for N episodes, reports standing metrics, saves height plot and video |
| `plot_runs.py` | compares all runs in `runs/` |

## Setup

The venv lives outside OneDrive so the 3 GB of PyTorch isn't synced:

```bash
C:\Users\hp\.venvs\humanoid\Scripts\activate
```

## Running

`python run_queue.py --max-parallel 1` runs the whole benchmark in priority order, which is
BAC → SAC → PPO → CrossQ → TD3 → RPO+sym. It only starts a job when at least 6 GB of commit is
free, because each CUDA process takes about 4.4 GB on Windows. Finished jobs get a
`runs/<name>/done` marker and are skipped on restart.

Individual runs:

```bash
python ppo.py --total-steps 10000000                 # PPO baseline (16 parallel envs)
python ppo.py --rpo-alpha 0.5 --sym-coef 0.02        # RPO + symmetric-action loss
python offpolicy.py --algo sac --total-steps 1000000
python offpolicy.py --algo td3 --total-steps 1000000
python offpolicy.py --algo crossq --total-steps 1000000
python offpolicy.py --algo bac --total-steps 2500000
tensorboard --logdir runs
python plot_runs.py
python evaluate.py runs/bac_s1/best.pt --episodes 10 --video
```

Each run writes to `runs/<name>/`: `train.csv`, `eval.csv`, `losses.csv`, TensorBoard events,
`best.pt` (best eval return) and `last.pt`.

## How to read the numbers

The return is almost entirely `torso_z / model.opt.timestep` summed over 1000 steps, so it is a
height readout. The metrics we track directly:

- **mean_height**: average torso height (m) over the episode. Upright torso is about 1.4 m.
- **final_height**: average over the last 100 steps, which tells you where it ended up.
- **frac_standing**: fraction of steps with torso above 1.0 m (the criterion used in Gymnasium PR #1767).
- **posture**: final height > 1.0 m is "standing", 0.3–1.0 m is "sitting/kneeling", below that is "lying".

Verified on this install: `model.opt.timestep` = 0.003, so the per-step height reward is
`torso_z / 0.003`. Lying flat (z ≈ 0.1) gives about 35k per episode, which is also what a random
policy gets. Sitting up (z ≈ 0.4) gives about 125k, and fully upright (z ≈ 1.3–1.4) gives about 430k+.

## Per-method design

**Shared across all methods**
- **Actions:** 17 torques. Policies output in [-1, 1] and `RescaleAction` maps that to the env's ±0.4.
- **State:** the raw 348-dim v5 observation, normalized with running mean and std. Contact forces
  and inertia terms have very different scales.
- **Reward:** the env reward, unchanged.
- **Discount:** γ = 0.99.

**PPO.** Separate actor and critic networks, each a 256-256 tanh MLP with orthogonal init. The
log-std is a learned free parameter, not conditioned on state.
- Reward is normalized by the std of the discounted return.
- At the 1000-step truncation, V(s_final) is bootstrapped. The env never terminates, so treating
  truncation as death would teach a time-dependent value the network can't observe.
- GAE λ = 0.95, entropy coefficient 0, advantages recomputed every epoch.
- 16 envs × 256 steps per update, 10 epochs, 8 minibatches.

**RPO.** The same as PPO, except that during the update the Gaussian mean is shifted by
U(-0.5, 0.5). This keeps exploration from collapsing.

**Symmetric-action loss.** Adds `coef · ||μ_left − s·μ_right||²`. The mirror signs `s` are
computed from the MuJoCo joint axes reflected through the sagittal plane. It targets the
"pushes up with one arm" failure.

**SAC.** Tanh-Gaussian actor with state-dependent std, twin Q critics, Polyak target networks
(τ = 0.005) and auto-tuned α with a target entropy of −17.
- 256-256 ReLU networks, batch size 256, learning rate 3e-4.
- 10k uniform-random warmup steps.
- Reward scaled by 0.01, which only changes the numerics.

**TD3.** Deterministic actor with Gaussian exploration noise of 0.1.
- Target-policy smoothing: noise 0.2, clipped at 0.5.
- Actor and target networks are updated every 2nd step.

**CrossQ.** No target networks.
- The critic has Batch Renormalization on its input and hidden layers, and is wide (2048-2048).
- (s,a) and (s',a') are passed through the critic in one batch, so the BN statistics match.
- Adam with β1 = 0.5 and learning rate 1e-3.
- Actor updated every 3rd step, one update per environment step (UTD 1).

**BAC (the BEE operator).** Builds on SAC by adding V(s), trained by expectile regression
(τ = 0.7) on target-Q of *buffer* actions. That makes V an in-sample estimate of the best action
the agent has already tried.
- The critic target is `r + γ[λ·V(s') + (1−λ)·(min Q'(s',a') − α log π)]` with λ = 0.5.
- This keeps the value of "lucky" partial stand-ups from being washed out by the current
  (worse) policy, which is why BAC gets past the kneeling plateau that SAC gets stuck on.
- 512-unit networks, batch size 512.
