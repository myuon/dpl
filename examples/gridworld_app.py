"""
GridWorld RL Visualization App

強化学習エージェントの学習過程を可視化するStreamlitアプリ
"""

import copy
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from gridWorld import (
    AgentTrainer,
    GridWorld,
    MonteCarloAgent,
    QLearningAgent,
    RandomGenGridWorld,
    TdAgent,
    value_iter,
)


def train_with_history(
    env_class,
    agent_class,
    num_episodes: int,
    gamma: float,
    epsilon: float,
    alpha: float,
    snapshot_interval: int = 100,
    seed: int | None = None,
):
    """学習を実行し、Q値の履歴を返す"""
    # 環境とエージェントを作成
    if env_class == RandomGenGridWorld:
        env = env_class(seed=seed)
    else:
        env = env_class()

    agent = agent_class(env, gamma=gamma, epsilon=epsilon, alpha=alpha)

    history = []

    # エージェントタイプに応じたhookを設定
    if agent_class == MonteCarloAgent:
        trainer = AgentTrainer(
            env,
            agent,
            num_episodes=num_episodes,
            on_episode_start=lambda a, e: a.reset(),
            on_step=lambda a, s, act, r, ns, d: a.add(s, act, r),
            on_done=lambda a: a.update(),
            on_episode_end=lambda a, e, ep: (
                history.append(copy.deepcopy(dict(a.Q)))
                if (ep + 1) % snapshot_interval == 0
                else None
            ),
        )
    else:
        # TdAgent, QLearningAgent
        trainer = AgentTrainer(
            env,
            agent,
            num_episodes=num_episodes,
            on_step=lambda a, s, act, r, ns, d: a.update(s, act, r, ns, d),
            on_episode_end=lambda a, e, ep: (
                history.append(copy.deepcopy(dict(a.Q)))
                if (ep + 1) % snapshot_interval == 0
                else None
            ),
        )

    trainer.train()

    return env, agent, history


def compute_ground_truth(env, gamma: float = 0.9):
    """DP法（Value Iteration）で真のQ値を計算"""
    V = defaultdict(lambda: 0)
    pi, V = value_iter(V, env, gamma)

    # VからQを計算
    Q_true = {}
    original_state = env.state
    for s in env.states():
        if s in env.walls:
            continue
        for a in env.get_actions():
            env.state = s
            next_s, r, done = env.step(a)
            r = r if r is not None else 0.0
            if done:
                Q_true[(s, a)] = r
            else:
                Q_true[(s, a)] = r + gamma * V[next_s]
    env.state = original_state

    return Q_true, V, pi


def compute_mse(Q_agent: dict, Q_true: dict, env) -> float:
    """Q値のMSEを計算"""
    errors = []
    for s in env.states():
        if s in env.walls or s == env.goal:
            continue
        for a in env.get_actions():
            q_a = Q_agent.get((s, a), 0.0)
            q_t = Q_true.get((s, a), 0.0)
            errors.append((q_a - q_t) ** 2)
    return np.mean(errors) if errors else 0.0


def render_q_diff(env, Q_agent: dict, Q_true: dict):
    """Q値の差分を可視化"""
    fig, ax = plt.subplots(figsize=(6, 4))

    # 差分を計算
    diff_grid = np.zeros((env.height, env.width))
    for y in range(env.height):
        for x in range(env.width):
            state = (y, x)
            if state in env.walls:
                diff_grid[y, x] = np.nan
            else:
                diffs = []
                for a in env.get_actions():
                    q_a = Q_agent.get((state, a), 0.0)
                    q_t = Q_true.get((state, a), 0.0)
                    diffs.append(abs(q_a - q_t))
                diff_grid[y, x] = np.mean(diffs)

    # ヒートマップ表示
    im = ax.imshow(diff_grid, cmap="Reds")
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))

    # 壁のマーク
    for y in range(env.height):
        for x in range(env.width):
            if (y, x) in env.walls:
                ax.text(x, y, "#", ha="center", va="center", fontsize=12, color="gray")
            else:
                ax.text(
                    x,
                    y,
                    f"{diff_grid[y, x]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    fig.colorbar(im, ax=ax)
    ax.set_title("Q Value Difference (|Q_agent - Q_true|)")
    return fig


def plot_learning_curve(history: list, Q_true: dict, env, snapshot_interval: int):
    """学習曲線をプロット"""
    fig, ax = plt.subplots(figsize=(8, 4))

    mse_values = []
    episodes = []

    for i, Q in enumerate(history):
        mse = compute_mse(Q, Q_true, env)
        mse_values.append(mse)
        episodes.append((i + 1) * snapshot_interval)

    ax.plot(episodes, mse_values, "b-", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("MSE")
    ax.set_title("Learning Curve (MSE vs Episode)")
    ax.grid(True, alpha=0.3)

    return fig


def derive_v_pi_from_q(Q: dict, env):
    """QからVとpiを導出"""
    V = {}
    pi = {}

    for s in env.states():
        if s in env.walls:
            V[s] = 0
            continue

        action_values = {a: Q.get((s, a), 0.0) for a in env.get_actions()}
        V[s] = max(action_values.values())

        if s != env.goal:
            best_action = max(action_values, key=lambda a: action_values[a])
            pi[s] = {a: 1.0 if a == best_action else 0.0 for a in env.get_actions()}

    return V, pi


# =============================================================================
# Streamlit App
# =============================================================================

st.set_page_config(page_title="GridWorld RL Visualization", layout="wide")
st.title("GridWorld RL Visualization")

# サイドバー設定
st.sidebar.title("設定")

env_type = st.sidebar.selectbox(
    "環境",
    ["GridWorld (4x3)", "RandomGenGridWorld (10x10)"],
    key="env_type",
)

agent_type = st.sidebar.selectbox(
    "エージェント",
    ["Q-Learning", "TD", "Monte Carlo"],
    key="agent_type",
)

num_episodes = st.sidebar.slider(
    "エピソード数",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100,
    key="num_episodes",
)

gamma = st.sidebar.slider(
    "gamma (割引率)",
    min_value=0.0,
    max_value=1.0,
    value=0.9,
    step=0.05,
    key="gamma",
)

epsilon = st.sidebar.slider(
    "epsilon (探索率)",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.05,
    key="epsilon",
)

alpha = st.sidebar.slider(
    "alpha (学習率)",
    min_value=0.01,
    max_value=1.0,
    value=0.1,
    step=0.01,
    key="alpha",
)

snapshot_interval = st.sidebar.slider(
    "スナップショット間隔",
    min_value=10,
    max_value=500,
    value=100,
    step=10,
    key="snapshot_interval",
)

# 環境とエージェントのマッピング
ENV_MAP = {
    "GridWorld (4x3)": GridWorld,
    "RandomGenGridWorld (10x10)": RandomGenGridWorld,
}

AGENT_MAP = {
    "Q-Learning": QLearningAgent,
    "TD": TdAgent,
    "Monte Carlo": MonteCarloAgent,
}

# キャッシュキー
cache_key = f"{env_type}_{agent_type}_{num_episodes}_{gamma}_{epsilon}_{alpha}_{snapshot_interval}"

# 再計算ボタン
if st.sidebar.button("再計算"):
    if cache_key in st.session_state:
        del st.session_state[cache_key]
    st.rerun()

# 学習実行 (キャッシュ)
if cache_key not in st.session_state:
    with st.spinner("学習中..."):
        env, agent, history = train_with_history(
            ENV_MAP[env_type],
            AGENT_MAP[agent_type],
            num_episodes,
            gamma,
            epsilon,
            alpha,
            snapshot_interval,
            seed=42,
        )
        Q_true, V_true, pi_true = compute_ground_truth(env, gamma)
        st.session_state[cache_key] = {
            "env": env,
            "agent": agent,
            "history": history,
            "Q_true": Q_true,
            "V_true": V_true,
            "pi_true": pi_true,
        }

# キャッシュからデータ取得
data = st.session_state[cache_key]
env = data["env"]
history = data["history"]
Q_true = data["Q_true"]
V_true = data["V_true"]
pi_true = data["pi_true"]

if not history:
    st.error("履歴がありません。スナップショット間隔を確認してください。")
    st.stop()

# エピソード選択
st.subheader("エピソード選択")
col1, col2, col3 = st.columns([6, 1, 1])

with col1:
    episode_idx = st.slider(
        "エピソード",
        min_value=0,
        max_value=len(history) - 1,
        value=len(history) - 1,
        key="episode_idx",
    )

# アニメーション状態
if "animation_running" not in st.session_state:
    st.session_state.animation_running = False

with col2:
    if st.button("▶ 再生"):
        st.session_state.animation_running = True

with col3:
    if st.button("⏹ 停止"):
        st.session_state.animation_running = False

# アニメーション処理
if st.session_state.animation_running:
    placeholder = st.empty()
    for i in range(episode_idx, len(history)):
        if not st.session_state.animation_running:
            break
        st.session_state.episode_idx = i
        time.sleep(0.3)
        st.rerun()
    st.session_state.animation_running = False

# 現在のQ値
current_Q = history[episode_idx]
current_episode = (episode_idx + 1) * snapshot_interval

st.write(f"**Episode: {current_episode}**")

# 可視化
st.subheader("学習結果")
col1, col2 = st.columns(2)

with col1:
    st.write("**Q(s, a) 可視化**")
    fig_q = env.render_q(current_Q, show=False)
    st.pyplot(fig_q)
    plt.close(fig_q)

with col2:
    st.write("**V(s) & Policy 可視化**")
    V, pi = derive_v_pi_from_q(current_Q, env)
    fig_v = env.render_v_pi(V, pi, show=False)
    st.pyplot(fig_v)
    plt.close(fig_v)

# DP真値との比較
st.subheader("DP法（Value Iteration）との比較")

mse = compute_mse(current_Q, Q_true, env)
st.metric("Mean Squared Error (MSE)", f"{mse:.6f}")

col1, col2 = st.columns(2)

with col1:
    st.write("**真のQ値 (DP法)**")
    fig_q_true = env.render_q(Q_true, show=False)
    st.pyplot(fig_q_true)
    plt.close(fig_q_true)

with col2:
    st.write("**差分 (|Q_agent - Q_true|)**")
    fig_diff = render_q_diff(env, current_Q, Q_true)
    st.pyplot(fig_diff)
    plt.close(fig_diff)

# 学習曲線
st.subheader("学習曲線")
fig_curve = plot_learning_curve(history, Q_true, env, snapshot_interval)
st.pyplot(fig_curve)
plt.close(fig_curve)

# 真のV/pi
st.subheader("真の価値関数と方策 (DP法)")
fig_true = env.render_v_pi(V_true, pi_true, show=False)
st.pyplot(fig_true)
plt.close(fig_true)
