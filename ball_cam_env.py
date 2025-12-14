# ball_cam_env.py
import numpy as np
import cv2
import time
from ball_obs_stream import BallObsStream


def greedy_policy(state):
    """
    一个手写小策略：
      state = [visible, cx, cy, area, px, py]
      如果看见球，就朝球的方向移动一步；
      如果看不见球，就不动。
    返回的动作含义：
      0=不动, 1=左, 2=右, 3=上, 4=下
    """
    visible, cx, cy, area, px, py = state

    # 看不到球就不动
    if visible < 0.5:
        return 0

    dx = cx - px
    dy = cy - py

    # 简单阈值，避免在很小误差附近左右抖
    eps = 0.01

    # 优先横向对齐，再纵向（你也可以反过来）
    if abs(dx) > abs(dy):
        if dx > eps:
            return 2  # 右
        elif dx < -eps:
            return 1  # 左
        else:
            return 0  # 差得不多就不动
    else:
        if dy > eps:
            return 4  # 下
        elif dy < -eps:
            return 3  # 上
        else:
            return 0







class BallCamEnv:
    """
    一个简单的“摄像头 + 虚拟守门员”环境：
      - 从 BallObsStream 读取真实球的位置 (cx, cy)
      - 屏幕上有一个虚拟守门员点 (px, py)
      - 动作：0=不动, 1=左, 2=右, 3=上, 4=下
      - 奖励：球可见时，reward = -球和守门员的欧氏距离；球不可见时，reward = 0
    """

    def __init__(self, width=640, height=480, step_size=0.05):
        # 创建球观测流（会启动 cam_track_guard.py 子进程）
        self.stream = BallObsStream(
            python_exe="python",
            script_path="tools/cam_track_guard.py",
            model_path="runs/detect/train3/weights/best.pt",
            cam=0,
        )

        self.width = width
        self.height = height
        self.step_size = step_size

        # 虚拟守门员的归一化位置 (px, py) ∈ [0,1]
        self.agent_pos = np.array([0.5, 0.5], dtype=np.float32)

        # 最近一次球的观测 [visible, cx, cy, area]
        self.last_ball_obs = np.array([0.0, -1.0, -1.0, 0.0], dtype=np.float32)

        # 用于渲染
        self.window_name = "Virtual Goalkeeper"

    def _update_ball_obs(self):
        """从 BallObsStream 获取最新一条球的位置，如果超时则保持上一帧。"""
        obs = self.stream.get_latest(timeout=0.05)
        if obs is not None:
            self.last_ball_obs = obs

    def _get_state(self):
        """
        返回给 RL 的状态向量:
          [visible, cx, cy, area, px, py]
        注意：cx, cy, px, py 都是归一化到 [0,1] 的坐标
        """
        visible, cx, cy, area = self.last_ball_obs
        px, py = self.agent_pos
        return np.array([visible, cx, cy, area, px, py], dtype=np.float32)

    def reset(self):
        """重置环境：守门员随机初始化到视野内某处，拉一帧球的观测。"""
        # 在 [0.1, 0.9] 范围内随机，避免一上来就刚好在球附近
        self.agent_pos[0] = np.random.uniform(0.1, 0.9)
        self.agent_pos[1] = np.random.uniform(0.1, 0.9)

        self._update_ball_obs()
        return self._get_state()

    def step(self, action: int):
        """
        执行动作并返回 (state, reward, done, info)
        action: 0=不动, 1=左, 2=右, 3=上, 4=下
        """
        # 1) 更新守门员位置（简单离散移动）
        dx, dy = 0.0, 0.0
        if action == 1:   # 左
            dx = -self.step_size
        elif action == 2:  # 右
            dx = self.step_size
        elif action == 3:  # 上
            dy = -self.step_size
        elif action == 4:  # 下
            dy = self.step_size

        self.agent_pos[0] = np.clip(self.agent_pos[0] + dx, 0.0, 1.0)
        self.agent_pos[1] = np.clip(self.agent_pos[1] + dy, 0.0, 1.0)

        # 2) 更新球观测
        self._update_ball_obs()

        # 3) 计算奖励
        visible, cx, cy, area = self.last_ball_obs
        px, py = self.agent_pos

        # 先把 cx, cy 限制在 [0,1]，避免 EMA 把框抹到边界外
        cx = float(np.clip(cx, 0.0, 1.0))
        cy = float(np.clip(cy, 0.0, 1.0))

        if visible >= 0.5:
            dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            reward = -float(dist)  # 离得越近 reward 越接近 0
            ball_visible = True
        else:
            reward = 0.0
            ball_visible = False

        state = self._get_state()
        done = False  # 这个任务可以先设成无终止
        info = {"ball_visible": ball_visible}

        return state, reward, done, info

    def render(self):
        """
        在一个单独窗口里画出：
          - 球的位置（红点）
          - 守门员的位置（蓝点）
        返回 False 表示用户按下 q/ESC 退出。
        """
        visible, cx, cy, area = self.last_ball_obs
        px, py = self.agent_pos

        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 画球（红色）
        if visible >= 0.5 and 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0:
            bx = int(cx * self.width)
            by = int(cy * self.height)
            cv2.circle(canvas, (bx, by), 12, (0, 0, 255), -1)

        # 画守门员（蓝色）
        gx = int(px * self.width)
        gy = int(py * self.height)
        cv2.circle(canvas, (gx, gy), 10, (255, 0, 0), -1)

        # 文本信息
        cv2.putText(
            canvas,
            "Red: ball (camera)  Blue: agent (RL)",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(self.window_name, canvas)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord("q"):
            return False
        return True

    def close(self):
        """关闭子进程和窗口"""
        self.stream.close()
        cv2.destroyWindow(self.window_name)


def main():
    env = BallCamEnv()
    state = env.reset()
    print("Env reset, initial state:", state)

    try:
        while True:
            # 使用手写贪心策略，而不是随机策略
            action = greedy_policy(state)
            state, reward, done, info = env.step(action)
            print(f"state={state}, reward={reward:.3f}, visible={info['ball_visible']}")
            if not env.render():
                break
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
