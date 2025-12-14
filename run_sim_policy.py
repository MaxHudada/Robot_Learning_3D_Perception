# run_sim_policy.py
import numpy as np
import torch

from sim_ball_env import SimBallEnv
from train_ball_rl_sim import PolicyNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) 建环境
    env = SimBallEnv()

    # 2) 建策略网络并加载训练好的参数
    policy = PolicyNet().to(device)
    state_dict = torch.load("sim_ball_policy.pt", map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()

    # 3) 用“贪心执行”的方式跑一条长轨迹，看行为表现
    state = env.reset().astype(np.float32)
    step_counter = 0


    try:
        while True:
            s_tensor = torch.from_numpy(state).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = policy(s_tensor)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                action = int(np.argmax(probs))  # 贪心动作

            # 临时打印前 50 步的动作和概率，看看是不是一直选 0
            if step_counter < 50:
                print(f"probs={probs}, action={action}")
                step_counter += 1

            next_state, reward, done, info = env.step(action)
            next_state = next_state.astype(np.float32)

            print(
                f"reward={reward:.3f}, "
                f"dist_now={info['dist_now']:.3f}, dist_future={info['dist_future']:.3f}"
            )

            if not env.render(wait_ms=20):
                break

            state = next_state

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
