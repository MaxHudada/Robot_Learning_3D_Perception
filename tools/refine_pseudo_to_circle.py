import os, glob, shutil, argparse
import cv2, numpy as np

def yolo_to_xyxy(x,y,w,h,W,H):
    cx,cy,bw,bh = x*W, y*H, w*W, h*H
    return cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2

def xyxy_to_yolo(x1,y1,x2,y2,W,H):
    bw, bh = max(1.0, x2-x1), max(1.0, y2-y1)
    cx, cy = x1+bw/2, y1+bh/2
    return cx/W, cy/H, bw/W, bh/H

def hough_circle(img):
    H,W = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g,(9,9),1.5)
    cs = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2,
                          minDist=int(min(W,H)*0.35),
                          param1=120, param2=30,
                          minRadius=int(min(W,H)*0.06),
                          maxRadius=int(min(W,H)*0.55))
    if cs is None: return None
    cs = np.squeeze(cs)
    if cs.ndim==1: cs=np.expand_dims(cs,0)
    cx0, cy0 = W/2, H/2
    best,score=None,-1e9
    for x,y,r in cs:
        s = r - 0.15*np.hypot(x-cx0,y-cy0)
        if s>score: best,score=(float(x),float(y),float(r)),s
    return best

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src_img", default="data/real_pos_raw")
    ap.add_argument("--src_lbl", default="runs/pseudo/real_ball_v1/labels")
    ap.add_argument("--out",     default="data/real_ball_yolo")
    args=ap.parse_args()

    out_img = os.path.join(args.out, "images")
    out_lbl = os.path.join(args.out, "labels")
    os.makedirs(out_img, exist_ok=True); os.makedirs(out_lbl, exist_ok=True)

    kept=0
    for p in sorted(glob.glob(os.path.join(args.src_img, "*.*"))):
        img = cv2.imread(p)
        if img is None: continue
        H,W = img.shape[:2]
        name = os.path.splitext(os.path.basename(p))[0]
        src_txt = os.path.join(args.src_lbl, name+".txt")
        out_txt = os.path.join(out_lbl, name+".txt")

        circ = hough_circle(img)
        if circ is not None:
            x,y,r = circ
            x1,y1,x2,y2 = x-r,y-r,x+r,y+r
        else:
            cand=[]
            if os.path.exists(src_txt):
                for line in open(src_txt,"r",encoding="utf-8"):
                    ss=line.strip().split()
                    if len(ss)<5: continue
                    _,x0,y0,w0,h0 = map(float, ss[:5])
                    a,b,c,d = yolo_to_xyxy(x0,y0,w0,h0,W,H)
                    area=(c-a)*(d-b)/(W*H+1e-9); ar=(c-a)/(d-b+1e-9)
                    if area>=0.01 and 0.6<=ar<=1.6: cand.append((area,(a,b,c,d)))
            if cand:
                cand.sort(key=lambda t:t[0], reverse=True)
                x1,y1,x2,y2 = cand[0][1]
            else:
                centers=[]
                if os.path.exists(src_txt):
                    for line in open(src_txt,"r",encoding="utf-8"):
                        ss=line.strip().split()
                        if len(ss)<5: continue
                        _,x0,y0,w0,h0 = map(float, ss[:5])
                        a,b,c,d = yolo_to_xyxy(x0,y0,w0,h0,W,H)
                        centers.append([(a+c)/2,(b+d)/2])
                if len(centers)>=3:
                    pts=np.array(centers,np.float32)
                    (cx,cy),r=cv2.minEnclosingCircle(pts)
                    x1,y1,x2,y2=cx-r,cy-r,cx+r,cy+r
                else:
                    continue

        x1,y1=max(0,x1),max(0,y1); x2,y2=min(W-1,x2),min(H-1,y2)
        x,y,w,h = xyxy_to_yolo(x1,y1,x2,y2,W,H)
        with open(out_txt,"w",encoding="utf-8") as f:
            f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        shutil.copy2(p, os.path.join(out_img, os.path.basename(p)))
        kept+=1

    print(f"[DONE] refined {kept} images -> {args.out}")

if __name__=="__main__":
    main()
