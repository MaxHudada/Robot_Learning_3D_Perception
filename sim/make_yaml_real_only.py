txt = """
path: .
train:
  - data/real_ball_yolo/images
val:
  - data/real_ball_yolo/images
names:
  0: ball
"""
open("data/ball_real_only.yaml","w",encoding="utf-8").write(txt.strip()+"\n")
print("[DONE] wrote data/ball_real_only.yaml")
