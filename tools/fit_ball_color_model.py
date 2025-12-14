# tools/fit_ball_color_model.py
# 作用：从 data/real_pos_raw 前 K 张里用霍夫圆截取球体区域，累积 HSV 直方图，保存到 models/ball_color_hist.npz
import os, glob, cv2, numpy as np

SRC = r"data/real_pos_raw"
OUT = r"models/ball_color_hist.npz"
os.makedirs("models", exist_ok=True)

def hough_circle(img):
    H,W = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g,(9,9),1.5)
    cs = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2,
                          minDist=int(min(W,H)*0.35),
                          param1=120, param2=28,
                          minRadius=int(min(W,H)*0.06),
                          maxRadius=int(min(W,H)*0.55))
    if cs is None: return None
    cs = np.squeeze(cs)
    if cs.ndim==1: cs=np.expand_dims(cs,0)
    # 选半径大且居中的圆
    cx0, cy0 = W/2, H/2
    best,score=None,-1e9
    for x,y,r in cs:
        s = r - 0.15*np.hypot(x-cx0,y-cy0)
        if s>score: best,score=(float(x),float(y),float(r)),s
    return best

# 直方图设置（可调）
HB, SB, VB = 24, 16, 8  # H/S/V 的 bin 数
hist = np.zeros((HB, SB, VB), np.float32)
used = 0

for i,p in enumerate(sorted(glob.glob(os.path.join(SRC, "*.*")))):
    if i>=120: break  # 取前 120 张做颜色建模
    img = cv2.imread(p)
    if img is None: continue
    H,W = img.shape[:2]
    c = hough_circle(img)
    if c is None: continue
    x,y,r = c
    x1,y1,x2,y2 = int(max(0,x-r)), int(max(0,y-r)), int(min(W-1,x+r)), int(min(H-1,y+r))
    crop = img[y1:y2, x1:x2].copy()
    if crop.size == 0: continue

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # 圆形 mask，避免背景干扰
    hh, ww = hsv.shape[:2]
    m = np.zeros((hh,ww), np.uint8)
    cv2.circle(m, (ww//2, hh//2), int(min(ww,hh)//2*0.95), 255, -1)
    vals = hsv[m>0]
    if vals.size == 0: continue
    h = vals[:,0]*(HB/180.0)  # H: [0,180)
    s = vals[:,1]*(SB/256.0)
    v = vals[:,2]*(VB/256.0)
    h = np.clip(h,0,HB-1).astype(int)
    s = np.clip(s,0,SB-1).astype(int)
    v = np.clip(v,0,VB-1).astype(int)
    for hh1,ss1,vv1 in zip(h,s,v):
        hist[hh1,ss1,vv1]+=1
    used += 1

hist = hist / (hist.sum()+1e-9)
np.savez_compressed(OUT, hist=hist, HB=HB, SB=SB, VB=VB)
print(f"[DONE] color model from {used} crops -> {OUT}")
