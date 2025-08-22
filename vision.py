import cv2
import numpy as np
import mss
import win32api, win32con, win32gui, win32ui
import ctypes
import time
import threading

# ---------- Enable DPI awareness ----------
def enable_dpi_awareness():
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

enable_dpi_awareness()


# ---------- Overlay ----------
class Overlay:
    def __init__(self, monitor, transparent_color=(255, 0, 255)):
        self.left = monitor["left"]
        self.top = monitor["top"]
        self.width = monitor["width"]
        self.height = monitor["height"]

        self.tcol = transparent_color

        wc = win32gui.WNDCLASS()
        wc.hInstance = win32api.GetModuleHandle(None)
        wc.lpszClassName = "OverlayWindow" + str(self.left) + "_" + str(self.top)
        wc.lpfnWndProc = {win32con.WM_DESTROY: self._on_destroy}
        self.atom = win32gui.RegisterClass(wc)

        ex = win32con.WS_EX_LAYERED | win32con.WS_EX_TOPMOST | win32con.WS_EX_TRANSPARENT
        style = win32con.WS_POPUP
        self.hwnd = win32gui.CreateWindowEx(
            ex, self.atom, "overlay", style,
            self.left, self.top, self.width, self.height,
            0, 0, wc.hInstance, None
        )

        key = (self.tcol[2] << 16) | (self.tcol[1] << 8) | self.tcol[0]
        win32gui.SetLayeredWindowAttributes(self.hwnd, key, 255, win32con.LWA_COLORKEY)
        win32gui.ShowWindow(self.hwnd, win32con.SW_SHOW)

    def _on_destroy(self, *_):
        win32gui.PostQuitMessage(0)
        return 0

    def _get_dc(self):
        hdc = win32gui.GetDC(self.hwnd)
        return hdc, win32ui.CreateDCFromHandle(hdc)

    def clear(self):
        hdc, dc = self._get_dc()
        brush = win32ui.CreateBrush(win32con.BS_SOLID, win32api.RGB(*self.tcol), 0)
        dc.SelectObject(brush)
        dc.Rectangle((0, 0, self.width, self.height))
        dc.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, hdc)

    def draw_dot(self, x, y, radius=6, color=(0, 0, 0)):
        hdc, dc = self._get_dc()
        brush = win32ui.CreateBrush(win32con.BS_SOLID, win32api.RGB(*color), 0)
        dc.SelectObject(brush)
        dc.Ellipse((x - radius, y - radius, x + radius, y + radius))
        dc.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, hdc)


# ---------- Target detection ----------
def find_red_centers(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 120, 120], np.uint8)
    upper1 = np.array([10, 255, 255], np.uint8)
    lower2 = np.array([170, 120, 120], np.uint8)
    upper2 = np.array([180, 255, 255], np.uint8)
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in cnts:
        if cv2.contourArea(c) < 8:
            continue
        x, y, w, h = cv2.boundingRect(c)
        centers.append((x + w // 2, y + h // 2))
    return centers


# ---------- Main loop for one monitor ----------
def run_on_monitor(monitor, radius=6, smooth=0.35, max_fps=60):
    overlay = Overlay(monitor)
    sct = mss.mss()

    smoothed = []  # list of [x, y]

    min_dt = 1.0 / max_fps
    t_prev = 0.0

    while True:
        # Pump messages
        while win32gui.PumpWaitingMessages():
            pass

        # Timing
        t = time.perf_counter()
        if t - t_prev < min_dt:
            time.sleep(min_dt - (t - t_prev))
        t_prev = time.perf_counter()

        # Capture
        img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        centers = find_red_centers(frame)

        overlay.clear()

        if centers:
            new_smoothed = []

            # Copy old smoothed points so we can match
            unmatched_old = smoothed[:]

            for cx, cy in centers:
                match = None
                best_dist = 50  # max distance to consider same target
                for old in unmatched_old:
                    dist = (old[0] - cx) ** 2 + (old[1] - cy) ** 2
                    if dist < best_dist ** 2:
                        best_dist = dist ** 0.5
                        match = old

                if match is not None:
                    # Smooth movement
                    nx = (1 - smooth) * match[0] + smooth * cx
                    ny = (1 - smooth) * match[1] + smooth * cy
                    new_smoothed.append([nx, ny])
                    unmatched_old.remove(match)
                else:
                    # New dot
                    new_smoothed.append([cx, cy])

            smoothed = new_smoothed

            # Draw each smoothed dot
            for sx, sy in smoothed:
                overlay.draw_dot(int(sx), int(sy), radius=radius, color=(0, 0, 0))
        else:
            smoothed = []

        win32gui.UpdateWindow(overlay.hwnd)


# ---------- Run all monitors ----------
def run_all_monitors():
    sct = mss.mss()
    monitors = sct.monitors[1:]  # skip index 0 (virtual bounding box)
    threads = []

    for mon in monitors:
        t = threading.Thread(target=run_on_monitor, args=(mon,), daemon=True)
        t.start()
        threads.append(t)

    # Keep alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run_all_monitors()
