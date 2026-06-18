"""
Integration test: vectorized PPO/SAC/DQN configurations x N seeds on the
vectorized ContextualDecouplerEnv. Uses the new tensor-based buffers.

This is the vectorized counterpart of ``runner.py`` and is kept at *parity*
with it: in addition to reward curves it measures, per training update, the
correlation between each model's learned per-head *importance* weights and the
environment's ground-truth state-dependent credit assignment (context=0 ->
head 0 controls the reward; context=1 -> head 1 controls the reward).

Outputs (all under ./dependence_results/):
  - runs/vectorized_<config>_seed<s>/   tensorboard scalars
  - vectorized_results.json             raw reward + importance metrics
  - vectorized_reward_curves.png        reward learning curves (mean +/- std)
  - vectorized_importance_analysis.png  per-context head dominance / correlation
"""

import gymnasium as gym
import numpy as np
import torch as th
import torch
import os
import json
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from flexibuddiesrl.PG_new_buffer import PG
from flexibuddiesrl.SAC_new_buffer import SAC
from flexibuddiesrl.DQN_new_buffer import DQN
from flexibuddiesrl.buffers import RolloutBuffer, ReplayBuffer
from toy_env import ContextualDecouplerEnv

# -- Hyper-parameters ---------------------------------------------------------
N_ACTIONS = 5
OBS_DIM = 3  # [context, target_0, target_1]
TOTAL_STEPS = 100_000
NUM_ENVS = 16
BATCH_SIZE = 256
LR = 1e-3
SEEDS = [0, 1, 2]

# Cap on how many importance snapshots we keep per run (off-policy algorithms
# update every step, which would otherwise produce tens of thousands of points).
IMPORTANCE_LOG_EVERY = 1

CONFIGS = {
    "PPO_shared_nomix": dict(algo="PPO", mix_type=None),
    "PPO_VDN": dict(algo="PPO", mix_type="VDN"),
    "PPO_QMIX": dict(algo="PPO", mix_type="QMIX", mixer_dim=64),
    "SAC_Q": dict(algo="SAC", mode="Q"),
    "SAC_V": dict(algo="SAC", mode="V"),
    "DQN_shared_nomix": dict(algo="DQN", mix_type=None),
    "DQN_VDN": dict(algo="DQN", mix_type="VDN"),
    "DQN_QMIX": dict(algo="DQN", mix_type="QMIX", mixer_dim=64),
}


def _make_model(cfg, device):
    algo = cfg["algo"]
    common = dict(
        obs_dim=OBS_DIM,
        continuous_action_dim=0,
        discrete_action_dims=[N_ACTIONS, N_ACTIONS],
        device=device,
        hidden_dims=[64, 64],
        lr=LR,
    )
    if algo == "PPO":
        kw = common.copy()
        kw.update(dict(n_epochs=4, mini_batch_size=128, mix_type=cfg.get("mix_type")))
        return PG(**kw)
    elif algo == "SAC":
        kw = common.copy()
        kw.update(dict(mode=cfg.get("mode"), target_discrete_entropy_percentage=0.1))
        return SAC(**kw)
    elif algo == "DQN":
        kw = common.copy()
        kw["continuous_action_dims"] = kw.pop("continuous_action_dim")
        kw.update(dict(mix_type=cfg.get("mix_type"), n_c_action_bins=5))
        return DQN(**kw)
    raise ValueError(f"Unknown algo: {algo}")


# -- Importance analysis helpers ----------------------------------------------
# (Ported verbatim in spirit from runner.py so the two runners report the same
#  statistics and the JSON schemas are interchangeable.)
def true_importance(context):
    """Return ideal credit weight [w0, w1] for two action heads given context."""
    if context == 0:
        return np.array([1.0, 0.0])
    else:
        return np.array([0.0, 1.0])


def _pearson_r(a, b):
    """Pearson correlation; returns 0.0 if either array has zero std."""
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _importance_analysis(
    importance_raw, batch_contexts, writer, update_step, prefix="Importance"
):
    """Log detailed importance metrics given per-step importance and contexts.

    Args:
        importance_raw: np.ndarray [n_steps, n_heads] - per-step importance weights
        batch_contexts: np.ndarray [n_steps] - context value (0 or 1) for each step
        writer: TensorBoard SummaryWriter
        update_step: global training update number
        prefix: tag prefix for tensorboard

    Returns:
        dict with correlation / dominance metrics for the whole batch
    """
    n_steps, n_heads = importance_raw.shape
    true_imp = np.stack([true_importance(c) for c in batch_contexts], axis=0)

    # Overall per-step correlation between learned and ground-truth importance.
    overall_corr = _pearson_r(importance_raw, true_imp)
    writer.add_scalar(f"{prefix}/correlation_overall", overall_corr, update_step)

    for d in range(n_heads):
        writer.add_scalar(
            f"{prefix}/head_{d}_mean", importance_raw[:, d].mean(), update_step
        )

    ctx0_mask = batch_contexts == 0
    ctx1_mask = batch_contexts == 1

    metrics = {"overall_corr": overall_corr}

    if ctx0_mask.sum() > 10:
        imp_ctx0 = importance_raw[ctx0_mask]
        true_ctx0 = true_imp[ctx0_mask]
        corr_ctx0 = _pearson_r(imp_ctx0, true_ctx0)
        writer.add_scalar(f"{prefix}/corr_ctx0", corr_ctx0, update_step)
        writer.add_scalar(f"{prefix}/ctx0_head0_mean", imp_ctx0[:, 0].mean(), update_step)
        writer.add_scalar(f"{prefix}/ctx0_head1_mean", imp_ctx0[:, 1].mean(), update_step)
        writer.add_scalar(
            f"{prefix}/ctx0_head0_dominance",
            (imp_ctx0[:, 0] > imp_ctx0[:, 1]).mean(),
            update_step,
        )
        metrics["corr_ctx0"] = corr_ctx0
        metrics["ctx0_h0_dom"] = float((imp_ctx0[:, 0] > imp_ctx0[:, 1]).mean())

    if ctx1_mask.sum() > 10:
        imp_ctx1 = importance_raw[ctx1_mask]
        true_ctx1 = true_imp[ctx1_mask]
        corr_ctx1 = _pearson_r(imp_ctx1, true_ctx1)
        writer.add_scalar(f"{prefix}/corr_ctx1", corr_ctx1, update_step)
        writer.add_scalar(f"{prefix}/ctx1_head0_mean", imp_ctx1[:, 0].mean(), update_step)
        writer.add_scalar(f"{prefix}/ctx1_head1_mean", imp_ctx1[:, 1].mean(), update_step)
        writer.add_scalar(
            f"{prefix}/ctx1_head1_dominance",
            (imp_ctx1[:, 1] > imp_ctx1[:, 0]).mean(),
            update_step,
        )
        metrics["corr_ctx1"] = corr_ctx1
        metrics["ctx1_h1_dom"] = float((imp_ctx1[:, 1] > imp_ctx1[:, 0]).mean())

    return metrics


def _to_numpy(x):
    return x.detach().cpu().numpy() if th.is_tensor(x) else np.asarray(x)


def _importance_and_contexts(rl, *, buffer=None, samples=None, n_heads=2):
    """Return aligned (importance [N, n_heads], contexts [N]) for a training step.

    The two algorithm families expose importance in different layouts, but in
    both cases the importance entry i was produced from the same observation
    whose context determines the ground-truth credit assignment:

      * PPO mixers: importance_raw is [T, E, A, D]; the matching observations are
        the still-resident rollout-buffer entries buffer.observations[:T] with
        layout [T, E, A, obs_dim]. Both flatten over [T, E, A] in C-order, so the
        context is simply obs[..., 0].
      * DQN-QMIX: importance_raw is [B, n_heads] for the *sampled* mini-batch, so
        the context is samples.observations[:, 0].
    """
    imp = _to_numpy(rl["importance_raw"])
    if imp.ndim == 4:  # [T, E, A, D] (PPO mixer)
        T = imp.shape[0]
        imp = imp.reshape(-1, imp.shape[-1])
        ctx = _to_numpy(buffer.observations[:T, :, :, 0]).reshape(-1)
    elif imp.ndim == 2:  # [B, n_heads] (DQN-QMIX)
        ctx = _to_numpy(samples.observations)[:, 0].reshape(-1)
    else:
        imp = imp.reshape(-1, n_heads)
        ctx = None
    return imp[:, :n_heads], (ctx.astype(int) if ctx is not None else None)


def run_vectorized_experiment(cfg_name, cfg, seed):
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Running [{cfg_name}_seed{seed}] on {device}...")

    np.random.seed(seed)
    th.manual_seed(seed)

    log_dir = f"./dependence_results/runs/vectorized_{cfg_name}_seed{seed}"
    writer = SummaryWriter(log_dir)

    env = gym.vector.SyncVectorEnv(
        [lambda: ContextualDecouplerEnv() for _ in range(NUM_ENVS)]
    )

    agent = _make_model(cfg, device)
    is_ppo = cfg["algo"] == "PPO"

    if is_ppo:
        buffer = RolloutBuffer(
            buffer_size=2048 // NUM_ENVS,
            obs_shape=(OBS_DIM,),
            action_dim=2,
            device=device,
            n_envs=NUM_ENVS,
            n_agents=1,
            log_probs_dim=2,  # 2 discrete heads
            full_gpu=True,
        )
    else:
        buffer = ReplayBuffer(
            buffer_size=50_000,
            obs_shape=(OBS_DIM,),
            action_dim=2,
            device=device,
            n_envs=NUM_ENVS,
            n_agents=1,
            full_gpu=True,
        )

    obs, _ = env.reset(seed=seed)
    obs_t = th.as_tensor(obs, device=device).unsqueeze(1).float()

    steps_done, ep_num, updates = 0, 0, 0
    episode_rewards = np.zeros(NUM_ENVS)
    smoothed_reward = 0.0
    have_smoothed = False
    reward_curve, importance_history = [], []

    while steps_done < TOTAL_STEPS:
        agent.eval()
        with th.no_grad():
            act_dict = agent.train_actions(obs_t)
            d_actions = act_dict["discrete_actions"]  # [NUM_ENVS, 1, 2]
            if is_ppo:
                values = act_dict["values"]
                log_probs = act_dict["discrete_log_probs"]

        d_actions_np = (
            d_actions.cpu().numpy() if torch.is_tensor(d_actions) else d_actions
        )
        env_actions = d_actions_np.reshape(NUM_ENVS, 2).astype(int)
        obs_next, rewards, terminations, truncations, info = env.step(env_actions)
        obs_next_t = th.as_tensor(obs_next, device=device).unsqueeze(1).float()

        if is_ppo:
            buffer.add(
                obs=obs_t if buffer.full_gpu else obs[np.newaxis, ...],
                action=d_actions if buffer.full_gpu else d_actions_np,
                reward=rewards[np.newaxis, ...],
                termination=terminations[np.newaxis, ...],
                truncation=truncations[np.newaxis, ...],
                value=values,
                log_prob=log_probs,
            )
        else:
            buffer.add(
                obs=obs_t if buffer.full_gpu else obs[np.newaxis, ...],
                next_obs=obs_next_t if buffer.full_gpu else obs_next[np.newaxis, ...],
                action=d_actions if buffer.full_gpu else d_actions_np,
                reward=rewards[np.newaxis, ...],
                term=terminations[np.newaxis, ...],
                trunc=truncations[np.newaxis, ...],
            )

        episode_rewards += rewards
        for i in range(NUM_ENVS):
            if terminations[i] or truncations[i]:
                ep_r = float(episode_rewards[i])
                writer.add_scalar("Reward/episode", ep_r, ep_num)
                # EMA over episodes, mirroring runner.py's smoothing.
                if not have_smoothed:
                    smoothed_reward = ep_r
                    have_smoothed = True
                else:
                    smoothed_reward = 0.95 * smoothed_reward + 0.05 * ep_r
                writer.add_scalar("Reward/smoothed", smoothed_reward, ep_num)
                reward_curve.append((steps_done, smoothed_reward))
                ep_num += 1
                episode_rewards[i] = 0

        obs = obs_next
        obs_t = obs_next_t
        steps_done += NUM_ENVS

        # -- Training ----------------------------------------------------------
        rl = None
        samples = None
        if is_ppo and buffer.full:
            agent.train()
            with th.no_grad():
                last_values = agent.expected_V(obs_t)
            rl = agent.reinforcement_learn(
                buffer,
                last_values,
                terminations[np.newaxis, ...],
                truncations[np.newaxis, ...],
            )
            updates += 1
            # Importance must be read BEFORE the buffer is reset, while the
            # observations that produced it are still resident.
            if rl is not None and rl.get("importance_raw") is not None:
                imp, ctx = _importance_and_contexts(rl, buffer=buffer)
                if ctx is not None and updates % IMPORTANCE_LOG_EVERY == 0:
                    metrics = _importance_analysis(imp, ctx, writer, updates)
                    metrics["step"] = float(steps_done)
                    importance_history.append(metrics)
            for k in ["rl_actor_loss", "rl_critic_loss", "d_entropy", "joint_kl"]:
                if rl is not None and k in rl:
                    writer.add_scalar(f"RL/{k}", rl[k], updates)
            buffer.reset()

        elif not is_ppo and steps_done > 1000:
            agent.train()
            samples = buffer.sample(BATCH_SIZE)
            rl = agent.reinforcement_learn(samples)
            updates += 1
            if rl is not None and rl.get("importance_raw") is not None:
                imp, ctx = _importance_and_contexts(rl, samples=samples)
                if ctx is not None and updates % IMPORTANCE_LOG_EVERY == 0:
                    metrics = _importance_analysis(imp, ctx, writer, updates)
                    metrics["step"] = float(steps_done)
                    importance_history.append(metrics)
            for k in ["rl_loss", "critic_loss", "actor_loss"]:
                if rl is not None and k in rl:
                    writer.add_scalar(f"RL/{k}", rl[k], updates)

    env.close()
    writer.close()
    print(
        f"  [{cfg_name} seed={seed}] done - smoothed reward = {smoothed_reward:.2f}"
    )
    return {
        "smoothed_reward": float(smoothed_reward),
        "reward_curve": reward_curve,
        "importance_history": importance_history,
    }


# -- Reporting ----------------------------------------------------------------
def _print_reward_summary(all_results):
    print("\n" + "=" * 72)
    print(f"{'Config':<20} {'Seeds (smoothed reward)':>30}  {'Mean':>8}")
    print("-" * 72)
    for cfg_name, runs in all_results.items():
        scores = [r["smoothed_reward"] for r in runs]
        seeds_str = ", ".join(f"{s:.1f}" for s in scores)
        print(f"{cfg_name:<20} {seeds_str:>30}  {np.mean(scores):>8.1f}")
    print("=" * 72)


def _print_importance_summary(all_results):
    print("\n" + "=" * 72)
    print("IMPORTANCE / CREDIT-ASSIGNMENT ANALYSIS")
    print("  Ground truth: context=0 -> head_0 important;  context=1 -> head_1 important")
    print("-" * 72)
    for cfg_name, runs in all_results.items():
        any_history = any(len(r["importance_history"]) > 0 for r in runs)
        if not any_history:
            print(f"  {cfg_name:<18}  (no mixer - importance not tracked)")
            continue
        print(f"\n  {cfg_name}:")
        for si, r in enumerate(runs):
            hist = r["importance_history"]
            if len(hist) == 0:
                continue
            # Analyse the last 20% of training.
            tail = hist[max(0, len(hist) - len(hist) // 5):]
            if len(tail) == 0:
                tail = hist

            def _safe_mean(key, data=tail):
                vals = [h[key] for h in data if key in h]
                return np.mean(vals) if vals else float("nan")

            ov = _safe_mean("overall_corr")
            c0 = _safe_mean("corr_ctx0")
            c1 = _safe_mean("corr_ctx1")
            d0 = _safe_mean("ctx0_h0_dom")
            d1 = _safe_mean("ctx1_h1_dom")
            print(f"    seed {SEEDS[si]}:")
            print(f"      Overall correlation:  {ov:+.3f}")
            print(f"      Context=0  corr={c0:+.3f}  head0_dominance={d0:.1%}")
            print(f"      Context=1  corr={c1:+.3f}  head1_dominance={d1:.1%}")
    print("=" * 72)


def _interpolate_to_common_steps(curves, n_points=200):
    """Interpolate variable-length reward curves to a common step axis."""
    curves = [c for c in curves if len(c) > 0]
    if not curves:
        return None, None
    max_step = max(c[-1][0] for c in curves)
    common_steps = np.linspace(0, max_step, n_points)
    matrix = np.full((len(curves), n_points), np.nan)
    for i, curve in enumerate(curves):
        xs, ys = zip(*curve)
        matrix[i] = np.interp(common_steps, xs, ys)
    return common_steps, matrix


def _plot_reward_curves(all_results, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10.colors  # type: ignore
    for ci, (cfg_name, runs) in enumerate(all_results.items()):
        steps, matrix = _interpolate_to_common_steps(
            [r["reward_curve"] for r in runs]
        )
        if steps is None:
            continue
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
        color = colors[ci % len(colors)]
        ax.plot(steps, mean, label=cfg_name, color=color, linewidth=1.5)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.18, color=color)
    ax.set_xlabel("Environment Steps", fontsize=14)
    ax.set_ylabel("Smoothed Episode Reward", fontsize=14)
    ax.set_title(
        f"Vectorized Reward Curves (mean +/- 1 std, n={len(SEEDS)})", fontsize=14
    )
    ax.legend(fontsize=11, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"\n  Saved {path}")


def _plot_importance_analysis(all_results, path):
    mixer_cfgs = [
        (name, runs)
        for name, runs in all_results.items()
        if any(len(r["importance_history"]) > 0 for r in runs)
    ]
    if not mixer_cfgs:
        print("  (no mixer configs produced importance history - skipping figure)")
        return

    n_plots = len(mixer_cfgs)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5), squeeze=False)

    for pi, (cfg_name, runs) in enumerate(mixer_cfgs):
        ax = axes[0, pi]
        hists = [r["importance_history"] for r in runs if len(r["importance_history"]) > 0]
        alpha = 0.35 if len(hists) > 1 else 1.0
        for si, hist in enumerate(hists):
            upd = np.arange(len(hist))
            h0 = np.array([h.get("ctx0_h0_dom", np.nan) for h in hist])
            h1 = np.array([h.get("ctx1_h1_dom", np.nan) for h in hist])
            ov = np.array([h.get("overall_corr", np.nan) for h in hist])
            lbls = dict(
                a=("ctx=0 h0 dom" if si == 0 else None),
                b=("ctx=1 h1 dom" if si == 0 else None),
                c=("overall corr" if si == 0 else None),
            )
            ax.plot(upd, h0, color="C0", alpha=alpha, linewidth=0.8, label=lbls["a"])
            ax.plot(upd, h1, color="C1", alpha=alpha, linewidth=0.8, label=lbls["b"])
            ax.plot(upd, ov, color="C2", alpha=alpha, linewidth=0.8, label=lbls["c"])

        # Mean across seeds over the common (shortest) horizon.
        min_len = min(len(h) for h in hists)
        if min_len > 0:
            def _mean(key):
                return np.nanmean(
                    [[h.get(key, np.nan) for h in hist[:min_len]] for hist in hists],
                    axis=0,
                )

            upd = np.arange(min_len)
            ax.plot(upd, _mean("ctx0_h0_dom"), color="C0", linewidth=2.2,
                    label="mean h0 dom (ctx=0)")
            ax.plot(upd, _mean("ctx1_h1_dom"), color="C1", linewidth=2.2,
                    label="mean h1 dom (ctx=1)")
            ax.plot(upd, _mean("overall_corr"), color="C2", linewidth=2.2,
                    linestyle="--", label="mean overall corr")

        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="chance")
        ax.set_xlabel("Training Updates", fontsize=12)
        ax.set_ylabel("Dominance / Correlation", fontsize=12)
        ax.set_title(cfg_name, fontsize=12)
        ax.set_ylim(-0.15, 1.05)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Credit Assignment: Per-Context Head Dominance (n={len(SEEDS)})", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def _save_results(all_results, path):
    serialisable = {}
    for cfg_name, runs in all_results.items():
        serialisable[cfg_name] = []
        for r in runs:
            serialisable[cfg_name].append({
                "smoothed_reward": float(r["smoothed_reward"]),
                "reward_curve": [[float(s), float(v)] for s, v in r["reward_curve"]],
                "importance_history": [
                    {k: float(v) for k, v in h.items()} for h in r["importance_history"]
                ],
            })
    with open(path, "w") as f:
        json.dump(serialisable, f)
    print(f"\n  Saved experiment data to {path}")


def main():
    os.makedirs("./dependence_results", exist_ok=True)
    all_results = {}

    for cfg_name, cfg in CONFIGS.items():
        all_results[cfg_name] = []
        for seed in SEEDS:
            res = run_vectorized_experiment(cfg_name, cfg, seed)
            all_results[cfg_name].append(res)

    _print_reward_summary(all_results)
    _print_importance_summary(all_results)
    _save_results(all_results, "./dependence_results/vectorized_results.json")
    _plot_reward_curves(all_results, "./dependence_results/vectorized_reward_curves.png")
    _plot_importance_analysis(
        all_results, "./dependence_results/vectorized_importance_analysis.png"
    )


def debug_critic_learning(n_steps=50_000, batch_size=256, seed=0, probe_interval=5_000):
    """
    Fills a replay buffer with random actions and trains only the SAC V-mode critic.
    Periodically probes Q(s, a) at hand-picked states to check whether the critic
    is learning the right value function before the actor is involved at all.

    The key diagnostic is the Q-gap between optimal and wrong actions, which should
    converge to ~2.0 (the reward difference from a single correct vs. incorrect step).
    If the gap stays near 0, the critic has a bug. If the gap is ~2.0 but the agent
    still fails to learn, the problem is in the actor loss.
    """
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"\n=== Critic-Only Debug | SAC V mode | device={device} ===")

    rng = np.random.default_rng(seed)
    env = ContextualDecouplerEnv()

    agent = SAC(
        obs_dim=OBS_DIM,
        continuous_action_dim=0,
        discrete_action_dims=[N_ACTIONS, N_ACTIONS],
        device=device,
        hidden_dims=[64, 64],
        lr=LR,
        mode="V",
    )

    buffer = ReplayBuffer(
        buffer_size=50_000,
        obs_shape=(OBS_DIM,),
        action_dim=2,
        device=device,
        n_envs=1,
        n_agents=1,
        full_gpu=(device == "cuda"),
    )

    probes = [
        (np.array([0, 2, 3]), np.array([2, 0]), "ctx=0 t0=2 | a=(2,0) OPTIMAL"),
        (np.array([0, 2, 3]), np.array([3, 0]), "ctx=0 t0=2 | a=(3,0) WRONG  "),
        (np.array([1, 2, 3]), np.array([0, 3]), "ctx=1 t1=3 | a=(0,3) OPTIMAL"),
        (np.array([1, 2, 3]), np.array([0, 2]), "ctx=1 t1=3 | a=(0,2) WRONG  "),
    ]
    probe_obs = th.tensor(
        np.stack([p[0] for p in probes]), dtype=th.float32, device=device
    )
    probe_acts = th.tensor(
        np.stack([p[1] for p in probes]), dtype=th.long, device=device
    )
    probe_labels = [p[2] for p in probes]

    obs, _ = env.reset(seed=seed)
    update_count = 0
    critic_loss = float("nan")

    for step in range(1, n_steps + 1):
        action = env.action_space.sample()
        obs_next, reward, terminated, truncated, _ = env.step(action)

        buffer.add(
            obs=obs.astype(np.float32).reshape(1, 1, OBS_DIM),
            next_obs=obs_next.astype(np.float32).reshape(1, 1, OBS_DIM),
            action=np.array(action, dtype=np.float32).reshape(1, 1, 2),
            reward=np.array([[reward]], dtype=np.float32),
            term=np.array([[terminated]], dtype=np.float32),
            trunc=np.array([[truncated]], dtype=np.float32),
        )

        if terminated or truncated:
            obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
        else:
            obs = obs_next

        if step > 1000:
            agent.train()
            samples = buffer.sample(batch_size)
            rl = agent.reinforcement_learn(samples, critic_only=True)
            critic_loss = rl["critic_loss"]
            update_count += 1

        if step % probe_interval == 0:
            agent.eval()
            with th.no_grad():
                q_vals = agent.utility_function(probe_obs, actions=(probe_acts, None))
            q_vals = q_vals.cpu().tolist()
            print(
                f"\n[step={step:>6d}  updates={update_count:>5d}  critic_loss={critic_loss:.4f}]"
            )
            for label, q in zip(probe_labels, q_vals):
                print(f"  Q {label} = {q:+.4f}")
            gap_ctx0 = q_vals[0] - q_vals[1]
            gap_ctx1 = q_vals[2] - q_vals[3]
            print(f"  --> Q_gap ctx=0 (expect ~2.0): {gap_ctx0:+.4f}")
            print(f"  --> Q_gap ctx=1 (expect ~2.0): {gap_ctx1:+.4f}")

    env.close()
    print("\n=== Done ===")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "debug_critic":
        debug_critic_learning()
    elif len(sys.argv) > 1 and sys.argv[1] == "--plot-only":
        path = sys.argv[2] if len(sys.argv) > 2 else "./dependence_results/vectorized_results.json"
        print(f"Loading saved data from {path}...")
        with open(path) as f:
            all_results = json.load(f)
        # reward_curve entries come back as [step, value] lists.
        for runs in all_results.values():
            for r in runs:
                r["reward_curve"] = [tuple(p) for p in r["reward_curve"]]
        _print_reward_summary(all_results)
        _print_importance_summary(all_results)
        _plot_reward_curves(all_results, "./dependence_results/vectorized_reward_curves.png")
        _plot_importance_analysis(
            all_results, "./dependence_results/vectorized_importance_analysis.png"
        )
    else:
        main()
