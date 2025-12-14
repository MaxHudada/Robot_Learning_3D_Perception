# train_ball_rl_sim.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sim_ball_env import SimBallEnv


# ===== 1. 策略网络：输入 state(6 维)，输出 5 个动作的 logit =====
class PolicyNet(nn.Module):
    def __init__(self, state_dim=6, action_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ===== 2. 值函数网络：输入 state(6 维)，输出标量 V(s) =====
class ValueNet(nn.Module):
    def __init__(self, state_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)  # [B, 1]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    env = SimBallEnv()

    policy = PolicyNet().to(device)
    value_net = ValueNet().to(device)

    # 一个优化器同时管 actor + critic，简单粗暴够用
    optimizer = optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()),
        lr=1e-3,
    )

    gamma = 0.95          # 折扣因子
    critic_coef = 0.5     # 值函数 loss 的权重
    entropy_coef = 0.01   # 熵正则权重，防止动作概率过早塌缩

    max_episodes = 500
    max_steps_per_episode = 80

    try:
        for ep in range(max_episodes):
            state = env.reset().astype(np.float32)

            ep_reward_sum = 0.0
            last_loss = 0.0

            for t in range(max_steps_per_episode):
                s_tensor = torch.from_numpy(state).unsqueeze(0).to(device)  # [1,6]

                # ---- 前向：策略 + 值函数 ----
                logits = policy(s_tensor)               # [1,5]
                probs = torch.softmax(logits, dim=-1)   # [1,5]
                dist = torch.distributions.Categorical(probs)

                action = dist.sample()                  # [1]
                log_prob = dist.log_prob(action)        # [1]

                value = value_net(s_tensor).squeeze(1)  # [1]，V(s)

                # ---- 与环境交互一步 ----
                next_state, reward, done, info = env.step(int(action.item()))
                next_state = next_state.astype(np.float32)

                ep_reward_sum += reward

                ns_tensor = torch.from_numpy(next_state).unsqueeze(0).to(device)

                with torch.no_grad():
                    next_value = value_net(ns_tensor).squeeze(1)  # [1]

                # ---- 计算 advantage（一步 TD 误差）----
                # td_target = r + gamma * V(s')
                reward_t = torch.tensor(reward, dtype=torch.float32, device=device)
                td_target = reward_t + gamma * next_value          # [1]
                advantage = td_target - value                      # [1]

                # ---- Actor-Critic 损失 ----
                # policy loss：-log π(a|s) * advantage
                actor_loss = -log_prob * advantage.detach()        # detach 避免adv梯度回流到V

                # value loss：(V(s) - td_target)^2
                critic_loss = advantage.pow(2)

                # 熵正则：鼓励策略在训练初期更“贪玩”一点，避免动作概率塌缩
                entropy = dist.entropy()                           # [1]

                loss = actor_loss + critic_coef * critic_loss - entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                last_loss = loss.item()

                # 可选渲染：每 50 个 episode 看一眼效果
                if ep % 50 == 0:
                    if not env.render(wait_ms=1):
                        raise KeyboardInterrupt

                state = next_state

            avg_reward = ep_reward_sum / max_steps_per_episode
            print(
                f"Episode {ep+1}/{max_episodes} | "
                f"sum_reward={ep_reward_sum:.3f}  avg_step_reward={avg_reward:.4f}  loss={last_loss:.4f}"
            )

        # 训练完成后，只保存策略网络，用于后面 run_sim_policy 或接摄像头
        torch.save(policy.state_dict(), "sim_ball_policy.pt")
        print("Training finished. Saved policy to sim_ball_policy.pt")

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
