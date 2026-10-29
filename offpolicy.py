"""Off-policy actor-critics for HumanoidStandup-v5 in one file: SAC, TD3, CrossQ and BAC.

They share the replay buffer, actor and training loop, and differ in the critic and its target:
  sac    twin Q + Polyak target nets, max-entropy (soft) target.                Haarnoja et al. 2018
  td3    twin Q + target nets, deterministic actor, target-policy smoothing,
         delayed actor updates.                                                  Fujimoto et al. 2018
  crossq twin wide Q with Batch Renormalization and NO target nets; (s,a) and (s',a') go through
         the critic in one joint batch so BN statistics match. UTD=1.            Bhatt et al. 2024
  bac    SAC + a V(s) net fit by expectile regression on Q of *buffer* actions (an in-sample max).
         Target = lambda * V(s') + (1 - lambda) * soft SAC target (the BEE operator), so "lucky"
         partial stand-ups stored in the buffer are not forgotten.               Ji et al. 2024

Usage:  python offpolicy.py --algo bac --total-steps 1000000 --seed 1
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import Logger, RunningMeanStd, evaluate, make_env, posture_stats, set_seed, torso_height

# Per-algorithm defaults, taken from each paper / reference implementation.
ALGO_DEFAULTS = {
    "sac": dict(hidden=256, critic_hidden=256, batch_size=256, lr=3e-4, adam_beta1=0.9, policy_delay=1),
    "td3": dict(hidden=256, critic_hidden=256, batch_size=256, lr=3e-4, adam_beta1=0.9, policy_delay=2),
    # CrossQ (sbx defaults): wide 2048-unit critic, lr 1e-3 with Adam beta1=0.5, actor every 3rd update.
    "crossq": dict(hidden=256, critic_hidden=2048, batch_size=256, lr=1e-3, adam_beta1=0.5, policy_delay=3),
    # BAC paper: 512-unit nets, batch 512, lambda=0.5, expectile tau=0.7.
    "bac": dict(hidden=512, critic_hidden=512, batch_size=512, lr=3e-4, adam_beta1=0.9, policy_delay=1),
}
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=list(ALGO_DEFAULTS), required=True)
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--run-name", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--learning-starts", type=int, default=10_000, help="uniform-random steps before learning")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005, help="Polyak rate for target nets (not used by crossq)")
    # Raw per-step reward is ~35-470 (torso_z / 0.003). Scaling keeps Q-values in the hundreds; with
    # auto-tuned alpha this only changes numerics, not the optimum.
    p.add_argument("--reward-scale", type=float, default=0.01)
    p.add_argument("--obs-norm", action=argparse.BooleanOptionalAction, default=True,
                   help="normalise observations with running mean/std (cfrc_ext/cinert have very different scales)")
    p.add_argument("--alpha-init", type=float, default=1.0)
    p.add_argument("--autotune", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--target-entropy", type=float, default=None, help="default: -action_dim")
    p.add_argument("--expl-noise", type=float, default=0.1, help="td3 exploration noise")
    p.add_argument("--policy-noise", type=float, default=0.2, help="td3 target smoothing noise")
    p.add_argument("--noise-clip", type=float, default=0.5)
    p.add_argument("--bee-lambda", type=float, default=0.5, help="bac: weight of the exploitation (V) target")
    p.add_argument("--expectile", type=float, default=0.7, help="bac: expectile used to fit V")
    p.add_argument("--brn-warmup", type=int, default=100_000, help="crossq: updates before Batch Renorm corrections kick in")
    p.add_argument("--eval-every", type=int, default=25_000)
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--log-every", type=int, default=5_000)
    p.add_argument("--resume", action="store_true", help="continue from runs/<run-name>/last.pt if it exists")
    p.add_argument("--init-from", default=None,
                   help="start a new run from another run's checkpoint (weights, optimiser, obs stats, step count)")
    p.add_argument("--posture-coef", type=float, default=0.0,
                   help="weight of the upright-spine bonus (common.PostureReward); 0 = stock reward")
    p.add_argument("--cuda-graph",action=argparse.BooleanOptionalAction, default=True,
                   help="capture update/act as CUDA graphs (3-5x faster: these nets are launch-overhead bound)")
    for k, v in ALGO_DEFAULTS["sac"].items():
        p.add_argument("--" + k.replace("_", "-"), type=type(v), default=None)
    args = p.parse_args()
    for k, v in ALGO_DEFAULTS[args.algo].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    args.run_name = args.run_name or f"{args.algo}_s{args.seed}"
    args.cuda_graph = args.cuda_graph and args.device.startswith("cuda")
    return args


# ---------------------------------------------------------------- networks

class VLinear(nn.Module):
    """n independent Linear layers evaluated with one batched matmul: x [n, B, i] -> [n, B, o]."""

    def __init__(self, n, i, o):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n, i, o))
        self.bias = nn.Parameter(torch.empty(n, 1, o))
        bound = 1 / math.sqrt(i)  # same init as nn.Linear
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return torch.baddbmm(self.bias, x, self.weight)


class BatchRenorm(nn.Module):
    """Batch Renormalization (Ioffe 2017), per ensemble member, as CrossQ uses in its critic.

    Plain BN while the `renorm` buffer is 0 (warm-up), then the r/d corrections that pull batch
    statistics towards the running ones. Eval mode always uses running statistics. The switch is a
    buffer rather than a Python branch so the update can be captured once as a CUDA graph.
    """

    def __init__(self, n, f, momentum=0.01, eps=1e-3, rmax=3.0, dmax=5.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n, 1, f))
        self.bias = nn.Parameter(torch.zeros(n, 1, f))
        self.register_buffer("running_mean", torch.zeros(n, 1, f))
        self.register_buffer("running_var", torch.ones(n, 1, f))
        self.register_buffer("renorm", torch.zeros(()))
        self.momentum, self.eps, self.rmax, self.dmax = momentum, eps, rmax, dmax

    def forward(self, x):
        if self.training:
            mean = x.mean(1, keepdim=True)
            var = x.var(1, keepdim=True, unbiased=False)
            std = (var + self.eps).sqrt()
            running_std = (self.running_var + self.eps).sqrt()
            r = (std / running_std).clamp(1 / self.rmax, self.rmax).detach()
            d = ((mean - self.running_mean) / running_std).clamp(-self.dmax, self.dmax).detach()
            r, d = torch.lerp(torch.ones_like(r), r, self.renorm), d * self.renorm
            xhat = (x - mean) / std * r + d
            with torch.no_grad():
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(var, self.momentum)
        else:
            xhat = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
        return xhat * self.weight + self.bias


class Critic(nn.Module):
    """Ensemble of n Q(s, a) MLPs run in parallel; returns [n, B]."""

    def __init__(self, obs_dim, act_dim, hidden, n=2, batch_renorm=False):
        super().__init__()
        norm = (lambda f: BatchRenorm(n, f)) if batch_renorm else (lambda f: nn.Identity())
        self.n = n
        self.net = nn.Sequential(
            norm(obs_dim + act_dim),
            VLinear(n, obs_dim + act_dim, hidden), norm(hidden), nn.ReLU(),
            VLinear(n, hidden, hidden), norm(hidden), nn.ReLU(),
            VLinear(n, hidden, 1),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], -1).unsqueeze(0).expand(self.n, -1, -1)
        return self.net(x).squeeze(-1)


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class Actor(nn.Module):
    """Tanh-squashed Gaussian policy (state-dependent std). With deterministic_policy=True (TD3) it
    is a plain tanh MLP."""

    def __init__(self, obs_dim, act_dim, hidden, deterministic_policy=False):
        super().__init__()
        self.deterministic_policy = deterministic_policy
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = None if deterministic_policy else nn.Linear(hidden, act_dim)

    def forward(self, obs, deterministic=False):
        h = self.trunk(obs)
        mu = self.mu(h)
        if self.deterministic_policy or deterministic:
            return torch.tanh(mu), None
        std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
        u = mu + std * torch.randn_like(mu)
        # log pi(a) = log N(u) - sum log(1 - tanh(u)^2), with the numerically stable form of the Jacobian.
        logp = (-0.5 * ((u - mu) / std) ** 2 - std.log() - 0.5 * math.log(2 * math.pi)).sum(-1)
        logp = logp - (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return torch.tanh(u), logp


def build_actor(args, obs_dim, act_dim):
    return Actor(obs_dim, act_dim, args.hidden, deterministic_policy=args.algo == "td3")


# ---------------------------------------------------------------- replay buffer

class ReplayBuffer:
    """Ring buffer of raw observations; normalisation happens on the GPU with the current statistics.

    next_obs is not stored separately (that would double the ~1.4 GB per 1M steps): it is the next
    slot's obs, except where an episode ended, whose final obs is kept in a small side table.
    """

    def __init__(self, capacity, obs_dim, act_dim):
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros((capacity, act_dim), np.float32)
        self.rew = np.zeros(capacity, np.float32)
        self.done = np.zeros(capacity, np.float32)  # true termination only (this env never terminates)
        self.ended = np.zeros(capacity, bool)  # terminated or truncated: next obs is in self.final_obs
        self.final_obs = {}
        self.capacity, self.ptr, self.size = capacity, 0, 0

    def add(self, obs, act, rew, next_obs, done, ended):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i], self.done[i], self.ended[i] = obs, act, rew, done, ended
        self.final_obs.pop(i, None)
        if ended:
            self.final_obs[i] = next_obs.copy()
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_into(self, out):
        """Gather a random batch straight into the given arrays (pinned staging memory)."""
        # Skip the newest transition: its successor slot isn't written yet.
        back = np.random.randint(1, self.size, len(out["r"]))
        idx = (self.ptr - 1 - back) % self.capacity
        np.take(self.obs, idx, axis=0, out=out["o"])
        np.take(self.act, idx, axis=0, out=out["a"])
        np.take(self.rew, idx, out=out["r"])
        np.take(self.done, idx, out=out["d"])
        np.take(self.obs, (idx + 1) % self.capacity, axis=0, out=out["o2"])
        for j in np.nonzero(self.ended[idx])[0]:
            out["o2"][j] = self.final_obs[idx[j]]


# ---------------------------------------------------------------- agent

def polyak(target, source, tau):
    with torch.no_grad():
        torch._foreach_lerp_(list(target.parameters()), list(source.parameters()), tau)


class Agent:
    """Holds the networks and runs updates. All update/act inputs live in fixed ("static") tensors so
    each step can be captured once as a CUDA graph and replayed with a single launch."""

    def __init__(self, args, obs_dim, act_dim):
        self.args, self.algo, dev = args, args.algo, args.device
        crossq = self.algo == "crossq"
        self.actor = build_actor(args, obs_dim, act_dim).to(dev)
        self.critic = Critic(obs_dim, act_dim, args.critic_hidden, batch_renorm=crossq).to(dev)
        adam = lambda params: torch.optim.Adam(params, lr=args.lr, betas=(args.adam_beta1, 0.999),
                                               capturable=args.cuda_graph)
        self.actor_opt, self.critic_opt = adam(self.actor.parameters()), adam(self.critic.parameters())
        if not crossq:
            self.critic_targ = Critic(obs_dim, act_dim, args.critic_hidden).to(dev)
            self.critic_targ.load_state_dict(self.critic.state_dict())
            self.critic_targ.requires_grad_(False)
        if self.algo == "td3":
            self.actor_targ = build_actor(args, obs_dim, act_dim).to(dev)
            self.actor_targ.load_state_dict(self.actor.state_dict())
            self.actor_targ.requires_grad_(False)
        else:
            self.log_alpha = torch.tensor(math.log(args.alpha_init), device=dev, requires_grad=args.autotune)
            self.alpha_opt = adam([self.log_alpha])
            self.target_entropy = args.target_entropy if args.target_entropy is not None else -float(act_dim)
        if self.algo == "bac":
            self.vnet = ValueNet(obs_dim, args.critic_hidden).to(dev)
            self.v_opt = adam(self.vnet.parameters())

        B = args.batch_size
        self.obs_mean = torch.zeros(obs_dim, device=dev)
        self.obs_std = torch.ones(obs_dim, device=dev)
        self.obs_clip = 10.0 if args.obs_norm else float("inf")
        self.batch = {"o": torch.zeros(B, obs_dim, device=dev), "a": torch.zeros(B, act_dim, device=dev),
                      "r": torch.zeros(B, device=dev), "o2": torch.zeros(B, obs_dim, device=dev),
                      "d": torch.zeros(B, device=dev)}
        pin = dev.startswith("cuda")
        self.staging = {k: torch.zeros(v.shape, pin_memory=pin) for k, v in self.batch.items()}
        self.staging_np = {k: v.numpy() for k, v in self.staging.items()}
        self.obs_in = torch.zeros(1, obs_dim, device=dev)
        self.graphs, self.stats, self.n_updates = {}, {}, 0

    # ---- plumbing

    def _run(self, key, fn):
        """Call fn eagerly, or capture it as a CUDA graph on first use and replay it afterwards."""
        if not self.args.cuda_graph:
            return fn()
        if key not in self.graphs:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):  # warm-up iterations, required before capture
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = fn()
            self.graphs[key] = (graph, out)
        graph, out = self.graphs[key]
        graph.replay()
        return out

    def norm(self, x):
        return ((x - self.obs_mean) / self.obs_std).clamp(-self.obs_clip, self.obs_clip)

    def set_obs_stats(self, rms):
        self.obs_mean.copy_(torch.as_tensor(rms.mean, dtype=torch.float32))
        self.obs_std.copy_(torch.as_tensor(np.sqrt(rms.var + 1e-8), dtype=torch.float32))

    def load_batch(self, rb):
        """Sample the batch for the next update into pinned memory and queue its upload."""
        rb.sample_into(self.staging_np)
        for k, v in self.batch.items():
            v.copy_(self.staging[k], non_blocking=True)

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    # ---- acting

    def act(self, raw_obs, deterministic=False):
        self.obs_in.copy_(torch.as_tensor(raw_obs, dtype=torch.float32)[None])

        @torch.no_grad()
        def fn():
            return self.actor(self.norm(self.obs_in), deterministic)[0]

        a = self._run(("act", deterministic), fn)[0].cpu().numpy()
        if self.algo == "td3" and not deterministic:
            a = np.clip(a + self.args.expl_noise * np.random.randn(*a.shape), -1, 1)
        return a

    # ---- learning

    def update(self):
        if self.algo == "crossq" and self.n_updates == self.args.brn_warmup:
            for m in self.critic.modules():
                if isinstance(m, BatchRenorm):
                    m.renorm.fill_(1.0)
        with_actor = self.n_updates % self.args.policy_delay == 0
        self.stats.update(self._run(("update", with_actor), lambda: self._update(with_actor)))
        self.n_updates += 1

    def _update(self, with_actor):
        args, gamma, b = self.args, self.args.gamma, self.batch
        o, a, r, o2, d = self.norm(b["o"]), b["a"], b["r"], self.norm(b["o2"]), b["d"]
        stats = {}

        # ---- critic ----
        with torch.no_grad():
            if self.algo == "td3":
                noise = (torch.randn_like(a) * args.policy_noise).clamp(-args.noise_clip, args.noise_clip)
                a2, logp2 = (self.actor_targ(o2)[0] + noise).clamp(-1, 1), None
            else:
                a2, logp2 = self.actor(o2)
        if self.algo == "crossq":
            # One joint forward so Batch Renorm sees (s,a) and (s',a') together; no target network.
            q_all = self.critic(torch.cat([o, o2]), torch.cat([a, a2]))
            q, q_next = q_all[:, : len(o)], q_all[:, len(o):].detach()
            next_v = q_next.min(0).values - self.alpha * logp2
        else:
            q = self.critic(o, a)
            with torch.no_grad():
                q_next = self.critic_targ(o2, a2).min(0).values
                next_v = q_next if self.algo == "td3" else q_next - self.alpha * logp2
                if self.algo == "bac":
                    # BEE operator: blend the in-sample "best buffer action" value with the soft target.
                    next_v = args.bee_lambda * self.vnet(o2) + (1 - args.bee_lambda) * next_v
        target = r + gamma * (1 - d) * next_v
        critic_loss = 0.5 * ((q - target.detach()) ** 2).mean(1).sum()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        stats["critic_loss"], stats["q_mean"] = critic_loss.detach(), q.mean().detach()

        # ---- V(s) by expectile regression on target-Q of buffer actions (BAC) ----
        if self.algo == "bac":
            with torch.no_grad():
                q_buf = self.critic_targ(o, a).min(0).values
            diff = q_buf - self.vnet(o)
            weight = torch.abs(args.expectile - (diff < 0).float())
            v_loss = (weight * diff**2).mean()
            self.v_opt.zero_grad(set_to_none=True)
            v_loss.backward()
            self.v_opt.step()
            stats["v_loss"] = v_loss.detach()

        # ---- actor (and alpha), every policy_delay critic updates ----
        if with_actor:
            self.critic.requires_grad_(False)
            self.critic.eval()  # matters for crossq: use running BN statistics
            pi, logp = self.actor(o)
            q_pi = self.critic(o, pi)
            if self.algo == "td3":
                actor_loss = -q_pi[0].mean()
            else:
                actor_loss = (self.alpha * logp - q_pi.min(0).values).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            self.critic.train()
            self.critic.requires_grad_(True)
            stats["actor_loss"] = actor_loss.detach()
            if self.algo == "td3":
                polyak(self.actor_targ, self.actor, args.tau)
            else:
                stats["entropy"] = -logp.mean().detach()
                if args.autotune:
                    alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
                    self.alpha_opt.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    self.alpha_opt.step()
                stats["alpha"] = self.alpha
        if self.algo != "crossq" and (self.algo != "td3" or with_actor):
            polyak(self.critic_targ, self.critic, args.tau)
        return stats

    def _modules(self):
        nets = {k: getattr(self, k) for k in ("actor", "critic", "critic_targ", "actor_targ", "vnet") if hasattr(self, k)}
        opts = {k: getattr(self, k + "_opt") for k in ("actor", "critic", "alpha", "v") if hasattr(self, k + "_opt")}
        return nets, opts

    def state_dict(self):
        nets, opts = self._modules()
        d = {k: v.state_dict() for k, v in nets.items()}
        d["optim"] = {k: v.state_dict() for k, v in opts.items()}
        d["n_updates"] = self.n_updates
        if hasattr(self, "log_alpha"):
            d["log_alpha"] = self.log_alpha.detach().cpu()
        return d

    def load_state_dict(self, d):
        nets, opts = self._modules()
        for k, net in nets.items():
            net.load_state_dict(d[k])
        for k, opt in opts.items():
            if k in d.get("optim", {}):  # checkpoints from before resume support have no optimizer state
                opt.load_state_dict(d["optim"][k])
        if hasattr(self, "log_alpha"):
            with torch.no_grad():
                self.log_alpha.copy_(d["log_alpha"])
        self.n_updates = d["n_updates"]
        if self.algo == "crossq" and self.n_updates >= self.args.brn_warmup:
            for m in self.critic.modules():
                if isinstance(m, BatchRenorm):
                    m.renorm.fill_(1.0)


# ---------------------------------------------------------------- training loop

def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")  # TF32 matmuls; matters for CrossQ's 2048-wide critic
    run_dir = os.path.join("runs", args.run_name)
    ckpt_path = os.path.join(run_dir, "last.pt")
    ckpt, resumed = None, False
    if args.resume and os.path.exists(ckpt_path):
        ckpt, resumed = torch.load(ckpt_path, map_location=args.device, weights_only=False), True
    elif args.init_from:
        ckpt = torch.load(args.init_from, map_location=args.device, weights_only=False)
        assert ckpt["algo"] == args.algo, f"--init-from is a {ckpt['algo']} checkpoint, not {args.algo}"
    start = ckpt["step"] if ckpt else 0
    logger = Logger(run_dir, {"algo": args.algo, **vars(args)}, resume_step=start if resumed else None)

    env, eval_env = make_env(posture_coef=args.posture_coef), make_env(posture_coef=args.posture_coef)
    env.action_space.seed(args.seed)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    agent = Agent(args, obs_dim, act_dim)
    rb = ReplayBuffer(min(args.buffer_size, args.total_steps), obs_dim, act_dim)
    obs_rms = RunningMeanStd((obs_dim,))
    best = -np.inf
    if ckpt:
        ckpt.setdefault("n_updates", ckpt["step"] - args.learning_starts)
        agent.load_state_dict(ckpt)
        if ckpt["obs_rms"] is not None:
            obs_rms.load_state_dict(ckpt["obs_rms"])
            agent.set_obs_stats(obs_rms)
        best_path = os.path.join(run_dir, "best.pt")
        if resumed and os.path.exists(best_path):
            best = torch.load(best_path, map_location="cpu", weights_only=False)["eval"]["return"]

    def act_eval(raw_obs):
        return agent.act(raw_obs, deterministic=True)

    obs, _ = env.reset(seed=args.seed + start)
    obs_rms.update(obs[None])
    ep_ret, ep_heights = 0.0, []
    t0, last_t, last_step = time.time(), time.time(), start
    # The replay buffer is not checkpointed (~1.4 GB): after a resume it is refilled with the restored
    # policy (not random actions) for at least 50k steps before updates restart. Updating on a few
    # thousand fresh samples overfits and visibly drops performance.
    learn_from = start + (max(args.learning_starts, 50_000) if ckpt else args.learning_starts)
    print(f"[{args.run_name}] {args.algo} on {args.device}"
          f"{f', starting from step {start:,}' if ckpt else ''}: {vars(args)}", flush=True)

    for step in range(start + 1, args.total_steps + 1):
        if step <= learn_from and not ckpt:
            action = env.action_space.sample()
        else:
            action = agent.act(obs)
        if step > learn_from:
            agent.update()  # queued on the GPU; runs while the CPU steps the simulator below
        next_obs, reward, term, trunc, _ = env.step(action)
        rb.add(obs, action, reward * args.reward_scale, next_obs, float(term), term or trunc)
        if args.obs_norm:
            obs_rms.update(next_obs[None])
            if step % 1000 == 0 or step == learn_from:
                agent.set_obs_stats(obs_rms)
        ep_ret += reward
        ep_heights.append(float(torso_height(next_obs)))
        obs = next_obs
        if term or trunc:
            logger.log("train", step, episode_return=ep_ret, **posture_stats(ep_heights))
            obs, _ = env.reset()
            ep_ret, ep_heights = 0.0, []
        if step >= learn_from:
            agent.load_batch(rb)

        if step % args.log_every == 0 and agent.stats:
            now = time.time()
            sps = (step - last_step) / (now - last_t)
            last_t, last_step = now, step
            logger.log("losses", step, sps=sps, **{k: v.item() for k, v in agent.stats.items()})

        if step % args.eval_every == 0 or step == args.total_steps:
            ev = evaluate(eval_env, act_eval, args.eval_episodes)
            logger.log("eval", step, **ev)
            ckpt = {"algo": args.algo, "args": vars(args), "step": step, "eval": ev,
                    "obs_rms": obs_rms.state_dict() if args.obs_norm else None, **agent.state_dict()}
            torch.save(ckpt, os.path.join(run_dir, "last.pt"))
            if ev["return"] > best:
                best = ev["return"]
                torch.save(ckpt, os.path.join(run_dir, "best.pt"))
            elapsed = time.time() - t0
            print(f"[{args.run_name}] step {step:>8,}  eval return {ev['return']:>9,.0f}  "
                  f"mean z {ev['mean_height']:.2f}  final z {ev['final_height']:.2f}  "
                  f"standing {ev['frac_standing']:.0%}  tilt {ev['final_tilt']:.0f} deg  ({(step - start) / elapsed:.0f} sps, {elapsed / 60:.1f} min)",
                  flush=True)
    logger.close()


if __name__ == "__main__":
    main()
