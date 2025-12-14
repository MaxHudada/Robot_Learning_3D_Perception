# run_ball_policy.py
import time
import numpy as np
import torch
import torch.nn as nn

from ball_cam_env import BallCamEnv
from train_ball_rl import PolicyNet  # 直接复用同一个网络结构


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) 建环境
    env = BallCamEnv()

    # 2) 建策略网络并加载训练好的参数
    policy = PolicyNet().to(device)
    state_dict = torch.load("ball_policy.pt", map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()

    # 3) 进入“纯执行模式”：每一步用贪心动作 + 渲染
    state = env.reset().astype(np.float32)

    try:
        while True:
            s_tensor = torch.from_numpy(state).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = policy(s_tensor)
                probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(probs, dim=-1).item()  # 贪心选动作

            next_state, reward, done, info = env.step(action)
            next_state = next_state.astype(np.float32)

            # 这里就只负责展示效果，不更新参数
            print(f"step reward={reward:.3f}, visible={info['ball_visible']}")

            if not env.render():   # 红点=球，蓝点=RL 守门员
                break

            state = next_state
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
