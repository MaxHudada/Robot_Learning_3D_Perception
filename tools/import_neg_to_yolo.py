# tools/import_neg_to_yolo.py
from pathlib import Path
import shutil

SRC = Path("data/real_neg_raw")
DST_IMG = Path("data/real_ball_yolo/images")
DST_LAB = Path("data/real_ball_yolo/labels")
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

DST_IMG.mkdir(parents=True, exist_ok=True)
DST_LAB.mkdir(parents=True, exist_ok=True)

# 先统计现有的 neg_ 前缀，避免重名
i = 0
for p in DST_IMG.glob("neg_*"):
    try:
        i = max(i, int(p.stem.split("_")[1]) + 1)
    except Exception:
        pass

moved = 0
skipped = 0
for src in SRC.iterdir():
    if src.suffix.lower() not in EXTS or not src.is_file():
        skipped += 1
        continue
    new_name = f"neg_{i:05d}{src.suffix.lower()}"
    dst_img = DST_IMG / new_name
    dst_lab = DST_LAB / (dst_img.stem + ".txt")
    shutil.copy2(src, dst_img)           # 复制负样本图片
    dst_lab.write_text("")               # 生成空标签文件（负样本关键点）
    i += 1
    moved += 1

print(f"[DONE] imported negatives: {moved}, skipped: {skipped}")
print(f"images in {DST_IMG}: {len(list(DST_IMG.glob('*')))}")
print(f"labels in {DST_LAB}: {len(list(DST_LAB.glob('*.txt')))}")
