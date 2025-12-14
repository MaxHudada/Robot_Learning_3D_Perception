# cam_ball_rl.py
import cv2
import time
import math
import argparse
import numpy as np
from collections import deque
from ultralytics import YOLO

import torch
import torch.nn as nn


# ---------- 颜色+圆度闸门（带高光裁剪） ----------
def pass_color_round_gate(bgr_roi,
                          blue_min=0.12, white_min=0.02,
                          round_max=0.22, clip_v=0.92):
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)

    # 忽略过曝像素（反光高亮会破坏颜色判定）
    vmask = hsv[:, :, 2] < int(clip_v * 255)

    # 蓝色（根据你球的色调做两段包围）
    blue1 = (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 135) & (hsv[:, :, 1] >= 60) & (hsv[:, :, 2] >= 50)
    blue2 = (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 140) & (hsv[:, :, 1] >= 40) & (hsv[:, :, 2] >= 80)
    blue_mask = (blue1 | blue2) & vmask

    # 白色（低饱和 + 高亮）
    white_mask = (hsv[:, :, 1] <= 35) & (hsv[:, :, 2] >= 170)

    total = bgr_roi.shape[0] * bgr_roi.shape[1]
    blue_ratio = float(np.count_nonzero(blue_mask)) / max(1, total)
    white_ratio = float(np.count_nonzero(white_mask)) / max(1, total)

    # 圆度 = 面积 / 最小外接圆面积，越接近1越圆
    gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    round_err = 1.0
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        (x, y), r = cv2.minEnclosingCircle(c)
        circ_area = math.pi * (r ** 2) if r > 1 else 1.0
        roundness = float(area) / circ_area
        round_err = abs(1.0 - roundness)

    ok = (blue_ratio >= blue_min) and (white_ratio >= white_min) and (round_err <= round_max)
    return ok, blue_ratio, white_ratio, round_err


# ---------- RL 策略网络（和 SimBallEnv 训练时一致） ----------
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


def parse_args():
    ap = argparse.ArgumentParser("Camera + YOLO + RL goalkeeper demo")

    # YOLO 基本参数
    ap.add_argument("--model", type=str, required=True, help="YOLO *.pt path")
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--iou", type=float, default=0.70)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--max_det", type=int, default=3)

    # 闸门阈值（跟 cam_track_guard 一致）
    ap.add_argument("--blue_min", type=float, default=0.12)
    ap.add_argument("--white_min", type=float, default=0.02)
    ap.add_argument("--round_max", type=float, default=0.22)
    ap.add_argument("--area_min", type=float, default=0.008, help="min box area ratio")
    ap.add_argument("--area_max", type=float, default=0.35, help="max box area ratio")
    ap.add_argument("--clip_v", type=float, default=0.92, help="ignore pixels with V>clip_v")

    # ROI（归一化 0~1）
    ap.add_argument("--roi", type=float, nargs=4, default=[0.00, 0.00, 1.00, 1.00],
                    help="x1 y1 x2 y2 in [0,1], crop detect region")

    # RL 相关
    ap.add_argument("--policy", type=str, default="sim_ball_policy.pt",
                    help="trained RL policy weights")
    ap.add_argument("--agent_step", type=float, default=0.05,
                    help="goalkeeper step size per frame (in [0,1] coords)")

    return ap.parse_args()


def draw_box(img, box, color, txt):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if txt:
        cv2.putText(img, txt, (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def main(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 摄像头 & YOLO
    cap = cv2.VideoCapture(opt.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("!! cannot open camera")
        return

    model = YOLO(opt.model)

    # 加载 RL 策略
    policy = PolicyNet().to(device)
    state_dict = torch.load(opt.policy, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()

    # RL 守门员状态（归一化到 0~1）
    agent_pos = np.array([0.5, 0.5], dtype=np.float32)  # 中间起步
    last_ball = None  # 上一帧的 (cx_norm, cy_norm)

    ema_box = None
    last_good = None
    hold_ttl = 0
    fps_q = deque(maxlen=30)

    roi_frac = opt.roi
    win_cam = "BallCam"
    win_rl = "RL-Goalkeeper"

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        H, W = frame.shape[:2]
        rx1 = int(roi_frac[0] * W); ry1 = int(roi_frac[1] * H)
        rx2 = int(roi_frac[2] * W); ry2 = int(roi_frac[3] * H)
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(W, rx2), min(H, ry2)

        roi_img = frame[ry1:ry2, rx1:rx2].copy()

        # ---------- YOLO 预测 ----------
        res = model.predict(
            source=roi_img, verbose=False, stream=False,
            imgsz=opt.imgsz, conf=opt.conf, iou=opt.iou, device=opt.device
        )[0]

        boxes = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clses = res.boxes.cls.cpu().numpy()
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                # 转回全图坐标
                x1 += rx1; x2 += rx1
                y1 += ry1; y2 += ry1
                boxes.append((x1, y1, x2, y2, float(confs[i]), int(clses[i])))

        boxes = sorted(boxes, key=lambda z: z[4], reverse=True)[:opt.max_det]

        passed = None
        pass_info = None

        # ---------- 颜色/圆度/面积闸门 ----------
        for (x1, y1, x2, y2, cf, clsid) in boxes:
            area_r = ((x2 - x1) * (y2 - y1)) / float(W * H)
            if area_r < opt.area_min or area_r > opt.area_max:
                continue

            x1i, y1i = max(0, int(x1)), max(0, int(y1))
            x2i, y2i = min(W, int(x2)), min(H, int(y2))
            crop = frame[y1i:y2i, x1i:x2i]
            if crop.size == 0:
                continue

            ok, br, wr, re = pass_color_round_gate(
                crop,
                blue_min=opt.blue_min,
                white_min=opt.white_min,
                round_max=opt.round_max,
                clip_v=opt.clip_v
            )

            if ok and last_good is not None:
                # 距离守门：离上一次中心太远则怀疑是误检
                gx1, gy1, gx2, gy2 = last_good
                gcx = 0.5 * (gx1 + gx2); gcy = 0.5 * (gy1 + gy2)
                cx = 0.5 * (x1 + x2); cy = 0.5 * (y1 + y2)
                if (cx - gcx) ** 2 + (cy - gcy) ** 2 > (180 ** 2):
                    ok = False

            if ok:
                passed = (x1, y1, x2, y2, cf)
                pass_info = (br, wr, re)
                break

        # ---------- 画原始灰框 ----------
        color_raw = (160, 160, 160)
        for (x1, y1, x2, y2, cf, clsid) in boxes:
            draw_box(frame, (x1, y1, x2, y2), color_raw, f"raw {cf:.2f}")

        ball_visible = False
        ball_cx_norm = 0.5
        ball_cy_norm = 0.5

        # ---------- EMA 平滑后的“真球框” ----------
        if passed is not None:
            x1, y1, x2, y2, cf = passed
            last_good = (x1, y1, x2, y2)
            hold_ttl = 6

            cx = 0.5 * (x1 + x2); cy = 0.5 * (y1 + y2)
            w = (x2 - x1); h = (y2 - y1)

            if ema_box is None:
                ema_box = np.array([cx, cy, w, h], dtype=np.float32)
            else:
                ema_box = 0.6 * ema_box + 0.4 * np.array([cx, cy, w, h], dtype=np.float32)

        else:
            if last_good is not None and hold_ttl > 0 and ema_box is not None:
                hold_ttl -= 1
            else:
                ema_box = None

        # 若有有效 EMA 框，则认为球可见
        if ema_box is not None:
            ecx, ecy, ew, eh = ema_box
            ex1 = int(ecx - ew / 2); ey1 = int(ecy - eh / 2)
            ex2 = int(ecx + ew / 2); ey2 = int(ecy + eh / 2)

            br, wr, re = pass_info if pass_info is not None else (0.0, 0.0, 1.0)
            draw_box(frame, (ex1, ey1, ex2, ey2), (40, 220, 40),
                     f"ball B{br:.2f} W{wr:.2f} R{re:.2f}")

            ball_cx_norm = float(np.clip(ecx / W, 0.0, 1.0))
            ball_cy_norm = float(np.clip(ecy / H, 0.0, 1.0))
            ball_visible = True

        # ROI 边框
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (80, 80, 80), 1)

        # ---------- RL：用训练好的策略控制守门员 ----------
        if ball_visible:
            if last_ball is None:
                vx, vy = 0.0, 0.0
            else:
                vx = ball_cx_norm - last_ball[0]
                vy = ball_cy_norm - last_ball[1]

            vx = float(np.clip(vx, -0.2, 0.2))
            vy = float(np.clip(vy, -0.2, 0.2))
            last_ball = (ball_cx_norm, ball_cy_norm)

            state = np.array(
                [ball_cx_norm, ball_cy_norm, vx, vy, agent_pos[0], agent_pos[1]],
                dtype=np.float32,
            )

            s_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = policy(s_tensor)
                probs = torch.softmax(logits, dim=-1)
                action = int(torch.argmax(probs, dim=-1).item())

            # 动作：0=不动, 1=左, 2=右, 3=上, 4=下
            dx = dy = 0.0
            step = opt.agent_step
            if action == 1:
                dx = -step
            elif action == 2:
                dx = step
            elif action == 3:
                dy = -step
            elif action == 4:
                dy = step

            agent_pos[0] = float(np.clip(agent_pos[0] + dx, 0.0, 1.0))
            agent_pos[1] = float(np.clip(agent_pos[1] + dy, 0.0, 1.0))

        # ---------- 画 RL 守门员 + 红球（2D 平面） ----------
        demo_size = 600
        demo = np.ones((demo_size, demo_size, 3), dtype=np.uint8) * 255

        # 画红球（如果有）
        if ball_visible:
            bx = int(ball_cx_norm * demo_size)
            by = int(ball_cy_norm * demo_size)
            cv2.circle(demo, (bx, by), 12, (0, 0, 255), -1)

        # 画蓝色守门员
        gx = int(agent_pos[0] * demo_size)
        gy = int(agent_pos[1] * demo_size)
        cv2.circle(demo, (gx, gy), 10, (255, 0, 0), -1)

        # 距离显示
        if ball_visible:
            dist = math.sqrt((ball_cx_norm - agent_pos[0]) ** 2 +
                             (ball_cy_norm - agent_pos[1]) ** 2)
            cv2.putText(demo, f"dist={dist:.3f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(demo, "NO BALL", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        # FPS
        dt = time.time() - t0
        fps_q.append(1.0 / max(1e-6, dt))
        fps = sum(fps_q) / len(fps_q)
        cv2.putText(frame, f"FPS {fps:.1f}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(win_cam, frame)
        cv2.imshow(win_rl, demo)

        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    main(args)
