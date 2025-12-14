# ball_obs_stream.py
import subprocess
import threading
import queue
import numpy as np
import sys
import time
from pathlib import Path


class BallObsStream:
    """
    启动 tools/cam_track_guard.py 作为子进程，
    持续读取其 stdout 中形如：
        OBS visible cx cy area
    的行，并提供 get_latest() 方法给 RL 使用。
    """

    def __init__(
        self,
        python_exe="python",
        script_path="tools/cam_track_guard.py",
        model_path="runs/detect/train3/weights/best.pt",
        cam=0,
    ):
        self.obs_queue = queue.Queue()
        self.process = None
        self._stop = False

        # 组装命令行（和你平时在 PowerShell 里跑的一样）
        script_path = str(Path(script_path))
        model_path = str(Path(model_path))

        self.cmd = [
            python_exe,
            script_path,
            "--model",
            model_path,
            "--cam",
            str(cam),
            "--imgsz",
            "960",
            "--conf",
            "0.20",
            "--iou",
            "0.70",
            "--max_det",
            "1",
            "--only_pass",
            "0",
            "--ema",
            "0.75",
            "--hold",
            "8",
            "--roi",
            "0",
            "0",
            "1",
            "1",
            "--near_px",
            "50",
        ]

        print("Launching cam_track_guard with command:")
        print(" ".join(self.cmd))

        # 启动子进程
        self.process = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # 开一个线程专门读 OBS 行
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self):
        """后台线程：持续读取 stdout，把 OBS 行解析后放入队列。"""
        assert self.process.stdout is not None
        for line in self.process.stdout:
            if self._stop:
                break

            line = line.strip()
            # 你在 cam_track_guard 里打印的是：
            # OBS visible cx cy area
            if not line.startswith("OBS"):
                # 其他调试信息直接忽略（也可以 print 出来）
                # print(line)
                continue

            parts = line.split()
            if len(parts) != 5:
                # 格式不对就跳过
                continue

            try:
                vals = list(map(float, parts[1:]))  # [visible, cx, cy, area]
                obs = np.array(vals, dtype=np.float32)
                self.obs_queue.put(obs)
            except ValueError:
                continue

        print("Reader loop finished")

    def get_latest(self, timeout=0.1):
        """
        获取最近一条观测：
          - 如果队列里有多条，取最后一条（防止积压）
          - 如果 timeout 时间内一直没新数据，返回 None
        """
        try:
            obs = self.obs_queue.get(timeout=timeout)
        except queue.Empty:
            return None

        # 把队列里剩下的都清掉，只保留最新一条
        while True:
            try:
                obs = self.obs_queue.get_nowait()
            except queue.Empty:
                break
        return obs

    def close(self):
        """关闭子进程和线程"""
        self._stop = True
        if self.process is not None:
            try:
                self.process.terminate()
            except Exception:
                pass


def main():
    stream = BallObsStream(
        python_exe="python",
        script_path="tools/cam_track_guard.py",
        model_path="runs/detect/train3/weights/best.pt",
        cam=0,
    )

    try:
        print("Start reading OBS for RL (Ctrl+C to stop)...")
        while True:
            obs = stream.get_latest(timeout=1.0)
            if obs is not None:
                visible, cx, cy, area = obs
                print(f"RL got obs: visible={visible:.0f}, cx={cx:.3f}, cy={cy:.3f}, area={area:.4f}")
            else:
                print("No new obs...")
            time.sleep(0.05)  # 模拟 RL 每步 20Hz 左右
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        stream.close()


if __name__ == "__main__":
    main()
