# sim_ball_env.py
import math
import time
import numpy as np
import cv2


class SimBallEnv:
    """
    纯仿真环境：
      - 红球：在 [0,1]x[0,1] 平面里运动，有位置 (cx, cy) 和速度 (vx, vy)
      - 蓝点：守门员，位置 (px, py)，由智能体控制，上下左右离散移动
      - 状态 state = [cx, cy, vx, vy, px, py]  共 6 维
      - 奖励：同时考虑“当前距离”和“若干步之后的预测距离”，鼓励提前扑球
    """

    def __init__(self):
        # 时间步长（概念上的），可以不用管具体单位
        self.dt = 1.0

        # 蓝点每步的移动步长（归一化坐标）
        self.agent_step = 0.05

        # 球速度的范围（归一化坐标每步）
        self.min_speed = 0.01
        self.max_speed = 0.04

        # 内部状态
        self.ball_pos = np.zeros(2, dtype=np.float32)  # [cx, cy]
        self.ball_vel = np.zeros(2, dtype=np.float32)  # [vx, vy]
        self.agent_pos = np.zeros(2, dtype=np.float32)  # [px, py]

        # 渲染相关
        self.win_name = "SimBallEnv"
        self.img_size = 400

    def _sample_ball(self):
        """随机初始化球的位置和速度（大致模拟你手的移动速度）"""
        # 球的位置保证不一上来就贴边
        cx = np.random.uniform(0.2, 0.8)
        cy = np.random.uniform(0.2, 0.8)
        self.ball_pos[:] = [cx, cy]

        # 随机一个速度方向 + 速度大小
        angle = np.random.uniform(0, 2 * math.pi)
        speed = np.random.uniform(self.min_speed, self.max_speed)
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        self.ball_vel[:] = [vx, vy]

    def _sample_agent(self):
        """随机初始化守门员位置"""
        px = np.random.uniform(0.1, 0.9)
        py = np.random.uniform(0.1, 0.9)
        self.agent_pos[:] = [px, py]

    def _update_ball(self):
        """更新虚拟球的位置，并在边界处做“弹墙”反射"""
        self.ball_pos += self.ball_vel * self.dt

        # 简单的弹墙逻辑：超出边界就反弹
        for i in range(2):
            if self.ball_pos[i] < 0.0:
                self.ball_pos[i] = -self.ball_pos[i]
                self.ball_vel[i] *= -1.0
            elif self.ball_pos[i] > 1.0:
                self.ball_pos[i] = 2.0 - self.ball_pos[i]
                self.ball_vel[i] *= -1.0

        # 再保险裁一遍
        self.ball_pos[:] = np.clip(self.ball_pos, 0.0, 1.0)

    def reset(self):
        """
        重置环境：
          - 随机球位置和速度
          - 随机守门员位置
          - 返回初始状态 state = [cx, cy, vx, vy, px, py]
        """
        self._sample_ball()
        self._sample_agent()
        return self._get_state()

    def _get_state(self):
        """构造 RL 用的状态向量"""
        cx, cy = self.ball_pos
        vx, vy = self.ball_vel
        px, py = self.agent_pos
        return np.array([cx, cy, vx, vy, px, py], dtype=np.float32)

    def step(self, action: int):
        """
        执行动作并返回 (state, reward, done, info)

        action 定义：
          0 = 不动
          1 = 左
          2 = 右
          3 = 上
          4 = 下
        """
        # 1) 先更新守门员位置
        dx, dy = 0.0, 0.0
        if action == 1:       # 左
            dx = -self.agent_step
        elif action == 2:     # 右
            dx = self.agent_step
        elif action == 3:     # 上
            dy = -self.agent_step
        elif action == 4:     # 下
            dy = self.agent_step

        self.agent_pos[0] = float(np.clip(self.agent_pos[0] + dx, 0.0, 1.0))
        self.agent_pos[1] = float(np.clip(self.agent_pos[1] + dy, 0.0, 1.0))

        # 2) 更新球的位置
        self._update_ball()

        # 3) 计算奖励：偏向“未来若干步”的位置
        cx, cy = self.ball_pos
        vx, vy = self.ball_vel
        px, py = self.agent_pos

        # 预测 k 步之后的球位置（匀速假设）
        k = 4
        cx_future = float(np.clip(cx + k * vx, 0.0, 1.0))
        cy_future = float(np.clip(cy + k * vy, 0.0, 1.0))

        dist_now = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        dist_future = math.sqrt((cx_future - px) ** 2 + (cy_future - py) ** 2)

        # 基础奖励：只惩罚距离，不再惩罚动作
        reward = - (0.3 * dist_now + 0.7 * dist_future)

        # 如果未来位置非常接近（例如 < 0.05），给一个明显的“扑到球”奖励
        if dist_future < 0.05:
            reward += 1.0

        state = self._get_state()
        done = False   # 这一版先不终止，由外部训练脚本控制 episode 长度
        info = {
            "dist_now": dist_now,
            "dist_future": dist_future,
        }

        return state, reward, done, info

    def render(self, wait_ms: int = 1):
        """
        简单渲染：画一个 400x400 白底图，红点 = 球，蓝点 = 守门员
        返回 False 表示用户按下 q / ESC 退出
        """
        img = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 255

        cx, cy = self.ball_pos
        px, py = self.agent_pos

        bx = int(cx * (self.img_size - 1))
        by = int(cy * (self.img_size - 1))
        ax = int(px * (self.img_size - 1))
        ay = int(py * (self.img_size - 1))

        # 红球
        cv2.circle(img, (bx, by), 8, (0, 0, 255), -1)
        # 蓝守门员
        cv2.circle(img, (ax, ay), 8, (255, 0, 0), -1)

        cv2.imshow(self.win_name, img)
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == 27 or key == ord('q'):
            return False
        return True

    def close(self):
        cv2.destroyWindow(self.win_name)


if __name__ == "__main__":
    # 简单随机策略测试一下环境有没有写对
    env = SimBallEnv()
    state = env.reset()
    try:
        while True:
            # 随机动一动：0~4 动作
            action = np.random.randint(0, 5)
            state, reward, done, info = env.step(action)
            print(f"state={state}, reward={reward:.3f}, "
                  f"dist_now={info['dist_now']:.3f}, dist_future={info['dist_future']:.3f}")
            if not env.render(20):
                break
    finally:
        env.close()
