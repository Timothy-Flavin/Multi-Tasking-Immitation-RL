"""
Integration test: 6 PPO configurations x 3 seeds on ContextualDecouplerEnv.

Configurations:
  1. independent    - Two separate single-head PPO models (no mixer)
  2. shared_nomix   - One PPO with 2 discrete heads, mixer=None
  3. VDN             - VDN mixer, no offline buffer
  4. VDN_offline     - VDN mixer, offline critic buffer
  5. QMIX            - QMIX mixer, no offline buffer
  6. QMIX_offline    - QMIX mixer, offline critic buffer

Logs to tensorboard under ./runs/<config>_seed<s>/
Tracks per-step and per-context correlation between learned importance
weights and the true environment credit assignment.
"""

import sys, os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from flexibuddiesrl.PG_stabalized import PG
from flexibuff import FlexibleBuffer
from torch.utils.tensorboard import SummaryWriter
from toy_env import ContextualDecouplerEnv

# -- Hyper-parameters ---------------------------------------------------------
N_ACTIONS = 5
OBS_DIM = 3  # [context, target_0, target_1]
TOTAL_STEPS = 50_000
BATCH_SIZE = 1024
MINI_BATCH_SIZE = 128
N_EPOCHS = 3
LR = 1e-3
SEEDS = [0, 1]

CONFIGS = {
    # "independent": dict(
    #     independent=True,
    # ),
    # "shared_nomix": dict(
    #     discrete_action_dims=[N_ACTIONS, N_ACTIONS],
    #     mix_type=None,
    #     offline_critic_buffer=False,
    #     advantage_type="gae",
    # ),
    "VDN": dict(
        discrete_action_dims=[N_ACTIONS, N_ACTIONS],
        mix_type="VDN",
        offline_critic_buffer=False,
    ),
    "VDN_offline": dict(
        discrete_action_dims=[N_ACTIONS, N_ACTIONS],
        mix_type="VDN",
        offline_critic_buffer=True,
    ),
    "QMIX": dict(
        discrete_action_dims=[N_ACTIONS, N_ACTIONS],
        mix_type="QMIX",
        offline_critic_buffer=False,
        mixer_dim=64,  # smaller than VDN: 128 is overkill for 2 agents, 3-dim state
    ),
    "QMIX_offline": dict(
        discrete_action_dims=[N_ACTIONS, N_ACTIONS],
        mix_type="QMIX",
        offline_critic_buffer=True,
        mixer_dim=64,
    ),
}


def _make_common_kwargs():
    return dict(
        obs_dim=OBS_DIM,
        continuous_action_dim=0,
        min_actions=np.zeros(1),
        max_actions=np.zeros(1),
        device="cpu",
        entropy_loss=0.01,
        ppo_clip=0.2,
        value_clip=0.2,
        norm_advantages=True,
        anneal_lr=0,
        orthogonal=True,
        std_type="stateless",
        clip_grad=True,
        mini_batch_size=MINI_BATCH_SIZE,
        action_clamp_type=None,
        n_epochs=N_EPOCHS,
        lr=LR,
        logit_reg=0.0,
        importance_schedule=[0.0, 1.0, 100],
        importance_from_grad=True,
        joint_kl_penalty=0.1,
        target_kl=0.1,
        gae_lambda=0.95,
        on_policy_mixer=True,
        mixer_dim=128,
        gamma=0.99,
    )


def _make_buffer(n_discrete_heads, cardinalities):
    """Create a FlexibleBuffer matching the env."""
    return FlexibleBuffer(
        num_steps=BATCH_SIZE + 10,
        n_agents=1,
        discrete_action_cardinalities=cardinalities,
        track_action_mask=False,
        path="/tmp/ppo_mix_test",
        name="mix_test",
        memory_weights=False,
        global_registered_vars={
            "global_rewards": (None, np.float32),
        },
        individual_registered_vars={
            "obs": ([OBS_DIM], np.float32),
            "obs_": ([OBS_DIM], np.float32),
            "discrete_log_probs": ([n_discrete_heads], np.float32),
            "continuous_log_probs": (None, np.float32),
            "discrete_actions": ([n_discrete_heads], np.int64),
            "continuous_actions": ([1], np.float32),
        },
    )


# -- Importance analysis helpers -----------------------------------------------
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


def _importance_analysis(importance_raw, batch_contexts, writer, update_step,
                         prefix="Importance"):
    """Log detailed importance metrics given per-step importance and contexts.

    Args:
        importance_raw: np.ndarray [n_steps, n_heads] - per-step importance weights
        batch_contexts: np.ndarray [n_steps] - context value (0 or 1) for each step
        writer: TensorBoard SummaryWriter
        update_step: global training update number
        prefix: tag prefix for tensorboard

    Returns:
        dict with correlation metrics
    """
    n_steps, n_heads = importance_raw.shape
    true_imp = np.stack([true_importance(c) for c in batch_contexts], axis=0)

    # Overall per-step correlation
    overall_corr = _pearson_r(importance_raw, true_imp)
    writer.add_scalar(f"{prefix}/correlation_overall", overall_corr, update_step)

    # Mean importance per head
    for d in range(n_heads):
        writer.add_scalar(
            f"{prefix}/head_{d}_mean", importance_raw[:, d].mean(), update_step
        )

    # Per-context correlation and mean importance
    ctx0_mask = batch_contexts == 0
    ctx1_mask = batch_contexts == 1

    metrics = {"overall_corr": overall_corr}

    if ctx0_mask.sum() > 10:
        imp_ctx0 = importance_raw[ctx0_mask]
        true_ctx0 = true_imp[ctx0_mask]
        corr_ctx0 = _pearson_r(imp_ctx0, true_ctx0)
        writer.add_scalar(f"{prefix}/corr_ctx0", corr_ctx0, update_step)
        writer.add_scalar(f"{prefix}/ctx0_head0_mean", imp_ctx0[:, 0].mean(),
                          update_step)
        writer.add_scalar(f"{prefix}/ctx0_head1_mean", imp_ctx0[:, 1].mean(),
                          update_step)
        # Head-0 dominance when context=0 (should be > 0.5)
        writer.add_scalar(
            f"{prefix}/ctx0_head0_dominance",
            (imp_ctx0[:, 0] > imp_ctx0[:, 1]).mean(), update_step,
        )
        metrics["corr_ctx0"] = corr_ctx0
        metrics["ctx0_h0_dom"] = float((imp_ctx0[:, 0] > imp_ctx0[:, 1]).mean())

    if ctx1_mask.sum() > 10:
        imp_ctx1 = importance_raw[ctx1_mask]
        true_ctx1 = true_imp[ctx1_mask]
        corr_ctx1 = _pearson_r(imp_ctx1, true_ctx1)
        writer.add_scalar(f"{prefix}/corr_ctx1", corr_ctx1, update_step)
        writer.add_scalar(f"{prefix}/ctx1_head0_mean", imp_ctx1[:, 0].mean(),
                          update_step)
        writer.add_scalar(f"{prefix}/ctx1_head1_mean", imp_ctx1[:, 1].mean(),
                          update_step)
        # Head-1 dominance when context=1 (should be > 0.5)
        writer.add_scalar(
            f"{prefix}/ctx1_head1_dominance",
            (imp_ctx1[:, 1] > imp_ctx1[:, 0]).mean(), update_step,
        )
        metrics["corr_ctx1"] = corr_ctx1
        metrics["ctx1_h1_dom"] = float((imp_ctx1[:, 1] > imp_ctx1[:, 0]).mean())

    return metrics


# -- Run one config/seed -------------------------------------------------------
def run_joint(config_name, cfg, seed, writer):
    """Run a single joint-model (non-independent) experiment."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    kw = _make_common_kwargs()
    kw.update(cfg)
    kw["name"] = config_name
    if kw.get("mix_type") is not None:
        kw["advantage_type"] = "qmix"
    model = PG(**kw)

    n_heads = len(cfg["discrete_action_dims"])
    buf = _make_buffer(n_heads, cfg["discrete_action_dims"])
    buf.reset()

    env = ContextualDecouplerEnv(n_actions=N_ACTIONS)
    obs, _ = env.reset(seed=seed)
    ep_reward = 0.0
    ep_num = 0
    smoothed_reward = 0.0
    updates = 0
    batch_contexts = []  # context per step in current batch

    # Accumulate importance metrics over training for final report
    importance_history = []

    for step in range(TOTAL_STEPS):
        obs_f = obs.astype(np.float32)
        act_dict = model.train_actions(obs_f, step=True)
        dact = act_dict["discrete_actions"]
        dlp = act_dict["discrete_log_probs"]

        env_action = dact.astype(int)
        obs_, reward, terminated, truncated, _ = env.step(env_action)
        ep_reward += reward
        batch_contexts.append(int(obs[0]))

        rv = {
            "global_rewards": float(reward),
            "obs": [obs_f.copy()],
            "obs_": [obs_.astype(np.float32).copy()],
            "discrete_log_probs": dlp.copy(),
            "continuous_log_probs": np.zeros(1, dtype=np.float32),
            "discrete_actions": [dact.copy()],
            "continuous_actions": [np.zeros(1, dtype=np.float32)],
        }
        buf.save_transition(terminated=terminated, registered_vals=rv)

        obs = obs_.copy()
        if terminated or truncated:
            writer.add_scalar("Reward/episode", ep_reward, ep_num)
            smoothed_reward = 0.95 * smoothed_reward + 0.05 * ep_reward
            writer.add_scalar("Reward/smoothed", smoothed_reward, ep_num)
            ep_num += 1
            ep_reward = 0.0
            obs, _ = env.reset()

        if buf.steps_recorded >= BATCH_SIZE:
            mb = buf.sample_transitions(
                idx=np.arange(0, BATCH_SIZE), as_torch=True, device="cpu"
            )
            rl = model.reinforcement_learn(mb, 0)
            updates += 1

            # Log scalar metrics
            for k in ["rl_actor_loss", "rl_critic_loss", "d_entropy", "joint_kl"]:
                if k in rl:
                    writer.add_scalar(f"RL/{k}", rl[k], updates)

            # Detailed importance analysis if mixer is active
            if "importance_raw" in rl and rl["importance_raw"] is not None:
                ctxs = np.array(batch_contexts[-BATCH_SIZE:])
                imp_raw = rl["importance_raw"]
                # importance_raw is [n_steps, n_heads] - align lengths
                n = min(len(ctxs), imp_raw.shape[0])
                metrics = _importance_analysis(imp_raw[:n], ctxs[:n], writer, updates)
                importance_history.append(metrics)
            elif "importance_per_dim" in rl and rl["importance_per_dim"] is not None:
                # Fallback: log per-head means
                imp = rl["importance_per_dim"]
                for d in range(len(imp)):
                    writer.add_scalar(f"Importance/head_{d}_mean", imp[d], updates)

            batch_contexts = []
            buf.reset()

    writer.close()
    print(f"  [{config_name} seed={seed}] done - smoothed reward = {smoothed_reward:.2f}")

    return {
        "smoothed_reward": smoothed_reward,
        "importance_history": importance_history,
    }


def run_independent(seed, writer):
    """Run two separate single-head PPO models (baseline independent learners)."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    kw0 = _make_common_kwargs()
    kw0["discrete_action_dims"] = [N_ACTIONS]
    kw0["advantage_type"] = "gae"
    kw0["name"] = "independent_h0"
    model0 = PG(**kw0)

    kw1 = _make_common_kwargs()
    kw1["discrete_action_dims"] = [N_ACTIONS]
    kw1["advantage_type"] = "gae"
    kw1["name"] = "independent_h1"
    model1 = PG(**kw1)

    buf0 = _make_buffer(1, [N_ACTIONS])
    buf1 = _make_buffer(1, [N_ACTIONS])
    buf0.reset()
    buf1.reset()

    env = ContextualDecouplerEnv(n_actions=N_ACTIONS)
    obs, _ = env.reset(seed=seed)
    ep_reward = 0.0
    ep_num = 0
    smoothed_reward = 0.0
    updates = 0

    for step in range(TOTAL_STEPS):
        obs_f = obs.astype(np.float32)

        ad0 = model0.train_actions(obs_f, step=True)
        ad1 = model1.train_actions(obs_f, step=True)

        a0 = int(ad0["discrete_actions"][0])
        a1 = int(ad1["discrete_actions"][0])
        env_action = np.array([a0, a1])

        obs_, reward, terminated, truncated, _ = env.step(env_action)
        ep_reward += reward

        obs_f_ = obs_.astype(np.float32)
        for mdl, buf, act_d in [
            (model0, buf0, ad0),
            (model1, buf1, ad1),
        ]:
            rv = {
                "global_rewards": float(reward),
                "obs": [obs_f.copy()],
                "obs_": [obs_f_.copy()],
                "discrete_log_probs": act_d["discrete_log_probs"].copy(),
                "continuous_log_probs": np.zeros(1, dtype=np.float32),
                "discrete_actions": [act_d["discrete_actions"].copy()],
                "continuous_actions": [np.zeros(1, dtype=np.float32)],
            }
            buf.save_transition(terminated=terminated, registered_vals=rv)

        obs = obs_.copy()
        if terminated or truncated:
            writer.add_scalar("Reward/episode", ep_reward, ep_num)
            smoothed_reward = 0.95 * smoothed_reward + 0.05 * ep_reward
            writer.add_scalar("Reward/smoothed", smoothed_reward, ep_num)
            ep_num += 1
            ep_reward = 0.0
            obs, _ = env.reset()

        if buf0.steps_recorded >= BATCH_SIZE:
            mb0 = buf0.sample_transitions(
                idx=np.arange(0, BATCH_SIZE), as_torch=True, device="cpu"
            )
            mb1 = buf1.sample_transitions(
                idx=np.arange(0, BATCH_SIZE), as_torch=True, device="cpu"
            )
            rl0 = model0.reinforcement_learn(mb0, 0)
            rl1 = model1.reinforcement_learn(mb1, 0)
            updates += 1

            for k in ["rl_actor_loss", "rl_critic_loss", "d_entropy"]:
                if k in rl0:
                    writer.add_scalar(f"RL_h0/{k}", rl0[k], updates)
                if k in rl1:
                    writer.add_scalar(f"RL_h1/{k}", rl1[k], updates)

            buf0.reset()
            buf1.reset()

    writer.close()
    print(f"  [independent seed={seed}] done - smoothed reward = {smoothed_reward:.2f}")
    return {"smoothed_reward": smoothed_reward, "importance_history": []}


# -- Main ----------------------------------------------------------------------
def main():
    results = {}
    for cfg_name, cfg in CONFIGS.items():
        results[cfg_name] = []
        for seed in SEEDS:
            tag = f"{cfg_name}_seed{seed}"
            log_dir = os.path.join(os.path.dirname(__file__), "runs", tag)
            writer = SummaryWriter(log_dir=log_dir)

            if cfg.get("independent"):
                res = run_independent(seed, writer)
            else:
                res = run_joint(cfg_name, cfg, seed, writer)
            results[cfg_name].append(res)

    # -- Print reward summary --------------------------------------------------
    print("\n" + "=" * 72)
    print(f"{'Config':<20} {'Seeds':>30}  {'Mean':>8}")
    print("-" * 72)
    for cfg_name, runs in results.items():
        scores = [r["smoothed_reward"] for r in runs]
        seeds_str = ", ".join(f"{s:.1f}" for s in scores)
        mean = np.mean(scores)
        print(f"{cfg_name:<20} {seeds_str:>30}  {mean:>8.1f}")
    print("=" * 72)

    # -- Print importance analysis ---------------------------------------------
    print("\n" + "=" * 72)
    print("IMPORTANCE / CREDIT-ASSIGNMENT ANALYSIS")
    print("  Ground truth: context=0 -> head_0 important;  context=1 -> head_1 important")
    print("-" * 72)
    for cfg_name, runs in results.items():
        any_history = any(len(r["importance_history"]) > 0 for r in runs)
        if not any_history:
            print(f"  {cfg_name:<18}  (no mixer - importance not tracked)")
            continue

        print(f"\n  {cfg_name}:")
        for si, r in enumerate(runs):
            hist = r["importance_history"]
            if len(hist) == 0:
                continue
            # Analyse last 20% of training
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


if __name__ == "__main__":
    main()
