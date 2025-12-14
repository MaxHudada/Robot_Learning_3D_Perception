# tools/vis_labels.py
import os, glob, cv2
IMG_DIR = r"data/real_ball_yolo/images"
LBL_DIR = r"data/real_ball_yolo/labels"
OUT     = r"runs/vis_labels"
os.makedirs(OUT, exist_ok=True)

imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.*")))[:60]
for p in imgs:
    img = cv2.imread(p); h,w = img.shape[:2]
    name = os.path.splitext(os.path.basename(p))[0]
    txt = os.path.join(LBL_DIR, name+".txt")
    if not os.path.exists(txt): continue
    with open(txt, "r", encoding="utf-8") as f:
        for line in f:
            cls, cx,cy,bw,bh = line.strip().split()
            cx,cy,bw,bh = map(float,[cx,cy,bw,bh])
            x1 = int((cx-bw/2)*w); y1 = int((cy-bh/2)*h)
            x2 = int((cx+bw/2)*w); y2 = int((cy+bh/2)*h)
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(img, "ball", (x1,max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.imwrite(os.path.join(OUT, os.path.basename(p)), img)
print("[DONE] wrote to", OUT)
