# tools/cam_track_guard.py
import cv2
import time
import math
import argparse
import numpy as np
from collections import deque
from ultralytics import YOLO
import os

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


def parse_args():
    ap = argparse.ArgumentParser("Camera detect + gate + EMA smoothing for a single soccer ball")

    # 基本
    ap.add_argument("--model", type=str, required=True, help="YOLO *.pt path")
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--iou", type=float, default=0.70)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--max_det", type=int, default=3)

    # 闸门阈值
    ap.add_argument("--blue_min", type=float, default=0.12)
    ap.add_argument("--white_min", type=float, default=0.02)
    ap.add_argument("--round_max", type=float, default=0.22)
    ap.add_argument("--area_min", type=float, default=0.008, help="min box area ratio")
    ap.add_argument("--area_max", type=float, default=0.35, help="max box area ratio")
    ap.add_argument("--clip_v", type=float, default=0.92, help="ignore pixels with V>clip_v")

    # 只显示通过闸门的目标
    ap.add_argument("--only_pass", type=int, default=0)

    # EMA/守门
    ap.add_argument("--ema", type=float, default=0.60, help="EMA系数(0-1)，越大越稳")
    ap.add_argument("--hold", type=int, default=6, help="未通过时，最多保留上次框的帧数")
    ap.add_argument("--near_px", type=int, default=180, help="与上次中心点的距离容忍(px)")

    # ROI（归一化 0~1）
    ap.add_argument("--roi", type=float, nargs=4, default=[0.00, 0.00, 1.00, 1.00],
                    help="x1 y1 x2 y2 in [0,1], crop detect region")

    # 可选 tracker（不推荐，默认不用；启用需 --use_track）
    ap.add_argument("--use_track", action="store_true")
    ap.add_argument("--tracker", type=str, default="botsort_ball.yaml")

    return ap.parse_args()


def draw_box(img, box, color, txt):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if txt:
        cv2.putText(img, txt, (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


# ---------- RL 用的状态提取函数（新加） ----------  # <<< RL
def extract_ball_state(box_xyxy, frame_shape):
    """
    给定当前帧中球的框(box_xyxy, 形式为 [x1,y1,x2,y2]) 和图像尺寸，
    返回归一化后的观测向量 [visible, cx, cy, area]：
      - visible: 1.0 表示当前有球框，0.0 表示当前帧无球框
      - cx, cy:  球中心在图像中的归一化坐标 ∈ [0,1]
      - area:    球框在整幅图中所占的相对面积（大致反映远近）
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box_xyxy

    x1 = float(x1)
    y1 = float(y1)
    x2 = float(x2)
    y2 = float(y2)

    cx = ((x1 + x2) / 2.0) / max(1.0, w)
    cy = ((y1 + y2) / 2.0) / max(1.0, h)
    area = ((x2 - x1) * (y2 - y1)) / max(1.0, (w * h))

    return np.array([1.0, cx, cy, area], dtype=np.float32)


def main(opt):
    cap = cv2.VideoCapture(opt.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("!! cannot open camera")
        return

    model = YOLO(opt.model)

    ema_box = None               # [cx, cy, w, h]
    last_good = None             # 上一次通过闸门的框 (xyxy)
    hold_ttl = 0                 # 保留计数
    fps_q = deque(maxlen=30)

    roi_frac = opt.roi  # [x1,y1,x2,y2]
    win_name = "BallCam"

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

        # 仅在 ROI 内做检测
        roi_img = frame[ry1:ry2, rx1:rx2].copy()

        # 预测（默认不走 track，稳定且兼容）
        res = model.predict(source=roi_img, verbose=False, stream=False,
                            imgsz=opt.imgsz, conf=opt.conf, iou=opt.iou, device=opt.device)[0]

        boxes = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clses = res.boxes.cls.cpu().numpy()
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                # 转回到全图坐标
                x1 += rx1; x2 += rx1; y1 += ry1; y2 += ry1
                boxes.append((x1, y1, x2, y2, float(confs[i]), int(clses[i])))

        # 选出候选（最高分前 opt.max_det）
        boxes = sorted(boxes, key=lambda z: z[4], reverse=True)[:opt.max_det]

        # RL：默认用最高置信度的 raw 框作为候选（不依赖颜色闸门）
        rl_box_xyxy = None
        if boxes:
            x1, y1, x2, y2, cf, clsid = boxes[0]
            rl_box_xyxy = (x1, y1, x2, y2)

        passed = None
        pass_info = None
        current_box_xyxy = None  # 当前帧用于 RL 的“最终球框”（EMA 后的）

        for (x1, y1, x2, y2, cf, clsid) in boxes:
            # 面积闸门
            area_r = ((x2 - x1) * (y2 - y1)) / float(W * H)
            if area_r < opt.area_min or area_r > opt.area_max:
                continue

            # 颜色+圆度闸门（在框内裁一块）
            x1i, y1i = max(0, int(x1)), max(0, int(y1))
            x2i, y2i = min(W, int(x2)), min(H, int(y2))
            crop = frame[y1i:y2i, x1i:x2i]
            if crop.size == 0:
                continue
            ok, br, wr, re = pass_color_round_gate(
                crop, blue_min=opt.blue_min, white_min=opt.white_min,
                round_max=opt.round_max, clip_v=opt.clip_v
            )

            # 距离守门：离上一次中心太远则先拒绝（防止突然跳到脸上）
            if ok and last_good is not None:
                gx1, gy1, gx2, gy2 = last_good
                gcx = 0.5 * (gx1 + gx2); gcy = 0.5 * (gy1 + gy2)
                cx = 0.5 * (x1 + x2); cy = 0.5 * (y1 + y2)
                if (cx - gcx) ** 2 + (cy - gcy) ** 2 > (opt.near_px ** 2):
                    ok = False

            if ok:
                passed = (x1, y1, x2, y2, cf)
                pass_info = (br, wr, re)
                break  # 取第一枚通过闸门的高分框

        # 画图 & EMA 平滑
        color_raw = (160, 160, 160)
        for (x1, y1, x2, y2, cf, clsid) in boxes:
            if opt.only_pass:
                # 只显示“绿框”时，不画原始灰框
                continue
            draw_box(frame, (x1, y1, x2, y2), color_raw, f"raw {cf:.2f}")

        if passed is not None:
            x1, y1, x2, y2, cf = passed
            # 更新 last_good & hold
            last_good = (x1, y1, x2, y2)
            hold_ttl = opt.hold

            # EMA
            cx = 0.5 * (x1 + x2); cy = 0.5 * (y1 + y2)
            w = (x2 - x1); h = (y2 - y1)
            if ema_box is None:
                ema_box = np.array([cx, cy, w, h], dtype=np.float32)
            else:
                ema_box = opt.ema * ema_box + (1.0 - opt.ema) * np.array([cx, cy, w, h], dtype=np.float32)

            ecx, ecy, ew, eh = ema_box
            ex1 = int(ecx - ew / 2); ey1 = int(ecy - eh / 2)
            ex2 = int(ecx + ew / 2); ey2 = int(ecy + eh / 2)

            br, wr, re = pass_info
            draw_box(frame, (ex1, ey1, ex2, ey2), (40, 220, 40),
                     f"ball {cf:.2f}  B{br:.2f} W{wr:.2f} R{re:.2f}")

            # 这一帧最终用于 RL 的球框（绿色 EMA 框）           # <<< RL
            current_box_xyxy = (ex1, ey1, ex2, ey2)                 # <<< RL
        else:
            # 未通过闸门：显示“红色跟随框”（若在保留期内）
            if last_good is not None and hold_ttl > 0 and ema_box is not None:
                ecx, ecy, ew, eh = ema_box
                ex1 = int(ecx - ew / 2); ey1 = int(ecy - eh / 2)
                ex2 = int(ecx + ew / 2); ey2 = int(ecy + eh / 2)
                draw_box(frame, (ex1, ey1, ex2, ey2), (20, 20, 230), "tracking-only")
                hold_ttl -= 1

                # 跟踪期内也视为“有球框”，供 RL 使用        # <<< RL
                current_box_xyxy = (ex1, ey1, ex2, ey2)             # <<< RL
            else:
                ema_box = None  # 完全丢失

        # 画 ROI 边框
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (80, 80, 80), 1)

        # ---------- 生成 RL 观测并输出（新加） ----------
        if current_box_xyxy is not None:
            # 优先使用通过闸门 + EMA 的平滑框
            obs = extract_ball_state(current_box_xyxy, frame.shape)
        elif rl_box_xyxy is not None:
            # 如果颜色/圆度闸门没通过，就退回到最高分 raw 框
            obs = extract_ball_state(rl_box_xyxy, frame.shape)
        else:
            # 当前帧完全没有检测到球：visible=0, cx,cy=-1, area=0
            obs = np.array([0.0, -1.0, -1.0, 0.0], dtype=np.float32)

        print(
            "OBS",
            float(obs[0]),
            float(obs[1]),
            float(obs[2]),
            float(obs[3]),
            flush=True,
        )
        # -----------------------------------------------

        # FPS
        dt = time.time() - t0
        fps_q.append(1.0 / max(1e-6, dt))
        fps = sum(fps_q) / len(fps_q)
        cv2.putText(frame, f"FPS {fps:.1f}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow(win_name, frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    # 仅当显式要求时尝试走 track（可选）
    if args.use_track:
        # 简单检查：若 yaml 不存在则退化为 predict
        if not os.path.exists(args.tracker):
            print(f"!! tracker yaml not found: {args.tracker}, fallback to predict mode")
            args.use_track = False
    main(args)

