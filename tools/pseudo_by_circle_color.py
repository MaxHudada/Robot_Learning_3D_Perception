# tools/pseudo_by_circle_color.py
# 用圆心/半径直接生成“整球”YOLO框；支持 --alpha 控制松紧；带进度条与可视化抽检。

import os, glob, cv2, numpy as np, shutil, argparse, time
try:
    from tqdm import tqdm
except:
    tqdm = lambda x, **k: x

def hough_circles(img):
    H, W = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (9,9), 1.5)
    cs = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2,
                          minDist=int(min(W,H)*0.30),
                          param1=120, param2=26,
                          minRadius=int(min(W,H)*0.06),
                          maxRadius=int(min(W,H)*0.60))
    if cs is None: return []
    cs = np.squeeze(cs)
    if cs.ndim == 1: cs = np.expand_dims(cs, 0)
    # 选最大圆（更稳）
    cs = sorted([(float(x),float(y),float(r)) for (x,y,r) in cs], key=lambda t:t[2], reverse=True)
    return cs[:1]

def xyxy_to_yolo(x1,y1,x2,y2,W,H):
    bw, bh = max(1.0, x2-x1), max(1.0, y2-y1)
    cx, cy = x1 + bw/2, y1 + bh/2
    return cx/W, cy/H, bw/W, bh/H

def main(a):
    os.makedirs(a.out_img, exist_ok=True)
    os.makedirs(a.out_lbl, exist_ok=True)
    if a.save_viz:
        os.makedirs(a.viz_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(a.src, "*.*")))
    keep, miss = 0, 0
    t0 = time.time()

    for p in tqdm(paths, desc="Labeling by circle", unit="img"):
        img = cv2.imread(p)
        if img is None: continue
        H, W = img.shape[:2]

        circles = hough_circles(img)
        if not circles:
            miss += 1
            continue

        x,y,r = circles[0]
        # 用正方形外接框，再乘 alpha 略放大
        s = r * np.sqrt(2) * a.alpha
        x1, y1 = int(max(0, x - s)), int(max(0, y - s))
        x2, y2 = int(min(W-1, x + s)), int(min(H-1, y + s))

        cx,cy,bw,bh = xyxy_to_yolo(x1,y1,x2,y2,W,H)
        name = os.path.splitext(os.path.basename(p))[0]
        with open(os.path.join(a.out_lbl, name + ".txt"), "w", encoding="utf-8") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        shutil.copy2(p, os.path.join(a.out_img, os.path.basename(p)))
        keep += 1

        if a.save_viz and keep <= a.viz_n:
            vis = img.copy()
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.circle(vis, (int(x),int(y)), int(r), (0,200,255), 2)
            cv2.putText(vis, f"r={int(r)}", (x1, max(0,y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imwrite(os.path.join(a.viz_dir, os.path.basename(p)), vis)

    print("\n=== Summary ===")
    print(f"Total: {len(paths)}, kept: {keep}, no-circle: {miss}, out_img: {a.out_img}, out_lbl: {a.out_lbl}")
    print(f"Time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"data/real_pos_raw", help="原始正样本目录")
    ap.add_argument("--out_img", default=r"data/real_ball_yolo/images", help="输出图片目录")
    ap.add_argument("--out_lbl", default=r"data/real_ball_yolo/labels", help="输出标签目录")
    ap.add_argument("--alpha", type=float, default=1.05, help="外接正方形放缩系数（>1 稍放大）")
    ap.add_argument("--save_viz", action="store_true", help="保存抽检可视化")
    ap.add_argument("--viz_dir", default=r"runs/pseudo/viz_circle", help="可视化目录")
    ap.add_argument("--viz_n", type=int, default=40, help="最多保存多少张可视化")
    args = ap.parse_args()
    main(args)
