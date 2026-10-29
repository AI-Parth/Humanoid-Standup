"""PPO for HumanoidStandup-v5 (CleanRL-style single file), with optional RPO and a symmetric-action loss.

  plain PPO   : --rpo-alpha 0 --sym-coef 0   (default)
  RPO         : --rpo-alpha 0.5              perturb the Gaussian mean by U(-a, a) when re-evaluating
                                             log-probs during the update, which keeps the policy from
                                             collapsing its exploration (Rahman & Xue 2022)
  + symmetry  : --sym-coef 0.02              penalise ||mu_left - sign * mu_right||^2 so the humanoid
                                             uses both arms/legs instead of pushing up on one side

Choices that matter on this env (see README): observation normalisation, return-based reward
normalisation (raw per-step reward is ~35-470), bootstrapping V(s_final) at the 1000-step truncation
(the env never terminates), and recomputing advantages every epoch (Tianshou's MuJoCo finding).

Usage:  python ppo.py --total-steps 10000000 --seed 1
"""
import argparse
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from common import (Logger, RunningMeanStd, evaluate, make_env, mirror_action_pairs, posture_stats, set_seed,
                    torso_height)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=10_000_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--run-name", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--async-envs", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--num-steps", type=int, default=256, help="rollout length per env per update")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=8)
    p.add_argument("--update-epochs", type=int, default=10)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--norm-reward", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--recompute-adv", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rpo-alpha", type=float, default=0.0)
    p.add_argument("--sym-coef", type=float, default=0.0)
    p.add_argument("--eval-every", type=int, default=250_000)
    p.add_argument("--eval-episodes", type=int, default=3)
    args = p.parse_args()
    if args.run_name is None:
        tag = "rpo" if args.rpo_alpha > 0 else "ppo"
        args.run_name = f"{tag}{'_sym' if args.sym_coef > 0 else ''}_s{args.seed}"
    return args


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    """Separate tanh MLPs for actor and critic; state-independent learned log-std."""

    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()

        def mlp(out, std):
            return nn.Sequential(layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
                                 layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
                                 layer_init(nn.Linear(hidden, out), std=std))

        self.critic = mlp(1, 1.0)
        self.actor_mean = mlp(act_dim, 0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x).squeeze(-1)

    def dist(self, x, rpo_alpha=0.0):
        mean = self.actor_mean(x)
        if rpo_alpha > 0:
            mean = mean + torch.empty_like(mean).uniform_(-rpo_alpha, rpo_alpha)
        return Normal(mean, self.actor_logstd.expand_as(mean).exp()), mean


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(4)
    dev = torch.device(args.device)
    run_dir = os.path.join("runs", args.run_name)
    logger = Logger(run_dir, {"algo": "ppo", **vars(args)})

    # SAME_STEP autoreset: on truncation the true final observation is in info["final_obs"] so we can
    # bootstrap from it, and the returned obs is already the first obs of the next episode.
    # Sync by default: one env step is ~0.35 ms, and on Windows every async worker re-imports torch
    # (~2 GB of committed memory each), which exhausts the page file with 16+ workers.
    vec_cls = gym.vector.AsyncVectorEnv if args.async_envs else gym.vector.SyncVectorEnv
    envs = vec_cls([make_env] * args.num_envs, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    eval_env = make_env()
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]
    pairs = mirror_action_pairs(eval_env)
    right_idx = torch.tensor([p[0] for p in pairs], device=dev)
    left_idx = torch.tensor([p[1] for p in pairs], device=dev)
    mirror_sign = torch.tensor([p[2] for p in pairs], device=dev)

    agent = Agent(obs_dim, act_dim, args.hidden).to(dev)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    obs_rms, ret_rms = RunningMeanStd((obs_dim,)), RunningMeanStd(())
    disc_ret = np.zeros(args.num_envs)

    T, N = args.num_steps, args.num_envs
    batch_size = T * N
    minibatch = batch_size // args.num_minibatches
    num_updates = args.total_steps // batch_size
    obs_buf = torch.zeros(T, N, obs_dim, device=dev)
    act_buf = torch.zeros(T, N, act_dim, device=dev)
    logp_buf = torch.zeros(T, N, device=dev)
    rew_buf = torch.zeros(T, N, device=dev)
    end_buf = torch.zeros(T, N, device=dev)
    val_buf = torch.zeros(T, N, device=dev)

    def normalize(raw):
        return torch.as_tensor(obs_rms.normalize(raw), device=dev)

    def act_eval(raw):
        with torch.no_grad():
            return agent.actor_mean(normalize(raw[None]))[0].cpu().numpy()

    def gae(values, next_value):
        adv = torch.zeros_like(rew_buf)
        last = torch.zeros(N, device=dev)
        for t in reversed(range(T)):
            nv = next_value if t == T - 1 else values[t + 1]
            nonterm = 1.0 - end_buf[t]
            delta = rew_buf[t] + args.gamma * nv * nonterm - values[t]
            last = delta + args.gamma * args.gae_lambda * nonterm * last
            adv[t] = last
        return adv

    raw_obs, _ = envs.reset(seed=args.seed)
    obs_rms.update(raw_obs)
    ep_ret, ep_heights = np.zeros(N), [[] for _ in range(N)]
    global_step, best, t0 = 0, -np.inf, time.time()
    next_eval = args.eval_every
    print(f"[{args.run_name}] PPO on {dev}, {num_updates} updates of {batch_size} steps: {vars(args)}")

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            opt.param_groups[0]["lr"] = args.lr * (1 - (update - 1) / num_updates)
        finished = []

        # ---------------- rollout ----------------
        for t in range(T):
            x = normalize(raw_obs)
            with torch.no_grad():
                d, _ = agent.dist(x)
                action = d.sample()
                logp_buf[t] = d.log_prob(action).sum(-1)
                val_buf[t] = agent.get_value(x)
            obs_buf[t], act_buf[t] = x, action
            raw_obs, reward, term, trunc, info = envs.step(action.cpu().numpy())
            global_step += N
            ended = term | trunc
            final_obs = raw_obs.copy()
            if ended.any():
                for i in np.where(ended)[0]:
                    final_obs[i] = info["final_obs"][i]

            ep_ret += reward
            for i in range(N):
                ep_heights[i].append(float(torso_height(final_obs[i])))
            if args.norm_reward:
                # Scale rewards by the std of the discounted return (what gym's NormalizeReward does).
                disc_ret = disc_ret * args.gamma + reward
                ret_rms.update(disc_ret)
                scaled = np.clip(reward / np.sqrt(ret_rms.var + 1e-8), -10, 10)
                disc_ret[ended] = 0.0
            else:
                scaled = reward
            scaled = torch.as_tensor(scaled, dtype=torch.float32, device=dev)
            trunc_only = trunc & ~term
            if trunc_only.any():
                # Time-limit truncation is not a real ending: add gamma * V(s_final) to the reward.
                with torch.no_grad():
                    v_final = agent.get_value(normalize(final_obs[trunc_only]))
                scaled[torch.as_tensor(trunc_only, device=dev)] += args.gamma * v_final
            rew_buf[t], end_buf[t] = scaled, torch.as_tensor(ended, dtype=torch.float32, device=dev)
            for i in np.where(ended)[0]:
                finished.append({"episode_return": ep_ret[i], **posture_stats(ep_heights[i])})
                ep_ret[i], ep_heights[i] = 0.0, []
            obs_rms.update(raw_obs)

        next_x = normalize(raw_obs)

        # ---------------- update ----------------
        b_obs, b_act = obs_buf.reshape(-1, obs_dim), act_buf.reshape(-1, act_dim)
        b_logp = logp_buf.reshape(-1)
        with torch.no_grad():
            adv = gae(val_buf, agent.get_value(next_x))
        clipfracs = []
        for epoch in range(args.update_epochs):
            if args.recompute_adv and epoch > 0:
                with torch.no_grad():
                    values = agent.get_value(obs_buf)
                    adv = gae(values, agent.get_value(next_x))
                    val_ref = values
            else:
                val_ref = val_buf
            b_adv = adv.reshape(-1)
            b_ret = (adv + val_ref).reshape(-1)
            perm = torch.randperm(batch_size, device=dev)
            for start in range(0, batch_size, minibatch):
                mb = perm[start:start + minibatch]
                d, mean = agent.dist(b_obs[mb], args.rpo_alpha)
                new_logp = d.log_prob(b_act[mb]).sum(-1)
                log_ratio = new_logp - b_logp[mb]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clipfracs.append(((ratio - 1).abs() > args.clip_coef).float().mean().item())
                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(-mb_adv * ratio,
                                    -mb_adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)).mean()
                v_loss = 0.5 * ((agent.get_value(b_obs[mb]) - b_ret[mb]) ** 2).mean()
                entropy = d.entropy().sum(-1).mean()
                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * entropy
                sym_loss = ((mean[:, left_idx] - mirror_sign * mean[:, right_idx]) ** 2).sum(-1).mean()
                if args.sym_coef > 0:
                    loss = loss + args.sym_coef * sym_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                opt.step()
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        elapsed = time.time() - t0
        logger.log("losses", global_step, pg_loss=pg_loss.item(), v_loss=v_loss.item(), entropy=entropy.item(),
                   approx_kl=approx_kl.item(), clipfrac=np.mean(clipfracs), sym_loss=sym_loss.item(),
                   std=agent.actor_logstd.exp().mean().item(), sps=global_step / elapsed)
        if finished:
            logger.log("train", global_step, **{k: np.mean([f[k] for f in finished]) for k in finished[0]})

        if global_step >= next_eval or update == num_updates:
            next_eval += args.eval_every
            ev = evaluate(eval_env, act_eval, args.eval_episodes)
            logger.log("eval", global_step, **ev)
            ckpt = {"algo": "ppo", "args": vars(args), "step": global_step, "eval": ev,
                    "obs_rms": obs_rms.state_dict(), "agent": agent.state_dict()}
            torch.save(ckpt, os.path.join(run_dir, "last.pt"))
            if ev["return"] > best:
                best = ev["return"]
                torch.save(ckpt, os.path.join(run_dir, "best.pt"))
            print(f"[{args.run_name}] step {global_step:>10,}  eval return {ev['return']:>9,.0f}  "
                  f"mean z {ev['mean_height']:.2f}  final z {ev['final_height']:.2f}  "
                  f"standing {ev['frac_standing']:.0%}  ({global_step / elapsed:.0f} sps, {elapsed / 60:.1f} min)",
                  flush=True)
    envs.close()
    logger.close()


if __name__ == "__main__":
    main()
