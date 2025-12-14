import cv2, os, argparse, time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="data/real_pos_raw", help="输出目录")
    ap.add_argument("--source", type=int, default=0)
    ap.add_argument("--width",  type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--n",      type=int, default=200, help="目标张数")
    ap.add_argument("--step",   type=int, default=2, help="每隔多少帧存一张")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cap = cv2.VideoCapture(args.source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.source)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    cnt, i = 0, 0
    print(f"[INFO] 保存到 {args.outdir}  目标: {args.n}  (q退出)")
    while True:
        ok, f = cap.read()
        if not ok: break
        i += 1
        if i % args.step == 0 and cnt < args.n:
            fn = os.path.join(args.outdir, f"{time.strftime('%Y%m%d_%H%M%S')}_{cnt:04d}.jpg")
            cv2.imwrite(fn, f); cnt += 1
        cv2.imshow("capture (q退出)", f)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        if cnt >= args.n: break

    cap.release(); cv2.destroyAllWindows()
    print(f"[DONE] {cnt} frames -> {args.outdir}")

if __name__ == "__main__":
    main()
