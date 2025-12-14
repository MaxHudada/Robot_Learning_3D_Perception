# train_ball_rl.py
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ball_cam_env import BallCamEnv  # 复用你刚才写好的环境


# ===== 1. 简单策略网络：输入 state(6 维)，输出 5 个动作的 logit =====
class PolicyNet(nn.Module):
    def __init__(self, state_dim=6, action_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def discount_rewards(rewards, gamma=0.98):
    """从后往前做折扣回报 G_t"""
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    env = BallCamEnv()
    policy = PolicyNet().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    max_episodes = 50           # 先少一点，看看趋势
    max_steps_per_episode = 80  # 每个 episode 步数（真实时间大约几秒）

    try:
        for ep in range(max_episodes):
            state = env.reset()
            state = state.astype(np.float32)

            log_probs = []
            rewards = []

            ep_reward_sum = 0.0

            for t in range(max_steps_per_episode):
                s_tensor = torch.from_numpy(state).unsqueeze(0).to(device)  # [1,6]
                logits = policy(s_tensor)                                   # [1,5]
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                # 使用采样方式选动作（标准 REINFORCE）
                action = dist.sample()             # 0~4
                log_prob = dist.log_prob(action)

                # 与环境交互一步
                next_state, reward, done, info = env.step(int(action.item()))
                next_state = next_state.astype(np.float32)

                log_probs.append(log_prob)
                rewards.append(reward)
                ep_reward_sum += reward

                # 每一步都渲染，让效果和贪心版本一样：两个窗口实时显示
                if not env.render():
                    # 如果你在任意窗口里按下 q 或 ESC，就中断训练
                    raise KeyboardInterrupt
                time.sleep(0.02)  # 控制一下刷新频率，不然会跑太快

                state = next_state

            # 一轮 episode 结束，做 REINFORCE 更新
            optimizer.zero_grad()
            returns = discount_rewards(rewards, gamma=0.98)
            if len(returns) > 1:
                # 简单做个归一化，稳定训练
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            log_probs_tensor = torch.stack(log_probs)  # [T]
            loss = -torch.sum(log_probs_tensor * returns.to(device))

            loss.backward()
            optimizer.step()

            avg_reward = ep_reward_sum / max_steps_per_episode
            print(
                f"Episode {ep+1}/{max_episodes} | "
                f"sum_reward={ep_reward_sum:.3f}  avg_step_reward={avg_reward:.4f}  loss={loss.item():.4f}"
            )

        # 训练全部结束后，保存策略参数
        torch.save(policy.state_dict(), "ball_policy.pt")
        print("Training finished.")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
