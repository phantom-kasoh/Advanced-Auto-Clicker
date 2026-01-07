import tkinter as tk
from tkinter import ttk
import random
import time

import win32api
import win32con
import win32gui


# ---------- Transparent overlay technique ----------
# We use a COLORKEY (magenta) as fully transparent via WS_EX_LAYERED.
COLORKEY_RGB = (255, 0, 255)
COLORKEY_HEX = "#ff00ff"


def get_virtual_screen_rect():
    """Return (left, top, right, bottom) of the full virtual desktop."""
    left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
    top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)
    width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
    height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
    return left, top, left + width, top + height


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def vk_down(vk: int) -> bool:
    """Global key state (works even if tkinter window isn't focused)."""
    return (win32api.GetAsyncKeyState(vk) & 0x8000) != 0


class EdgeKey:
    """Rising-edge detector for keys (press events)."""
    def __init__(self, vk: int):
        self.vk = vk
        self.was_down = False

    def pressed(self) -> bool:
        down = vk_down(self.vk)
        hit = down and not self.was_down
        self.was_down = down
        return hit


class VirtualCursorOverlayApp:
    def __init__(self):
        # --------------- Control window ---------------
        self.root = tk.Tk()
        self.root.title("Pure Virtual Cursor Overlay (Control)")
        self.root.geometry("430x230")

        self.status = tk.StringVar(value="Ready.")
        self.pos_var = tk.StringVar(value="x=?, y=?")
        self.score_var = tk.StringVar(value="Hits: 0   Misses: 0   Acc: 0.0%")

        # speed / target count
        self.speed_var = tk.IntVar(value=6)
        self.targets_var = tk.IntVar(value=10)

        self.hits = 0
        self.misses = 0

        self._build_control_ui()

        # --------------- Overlay window ---------------
        self.overlay = tk.Toplevel(self.root)
        self.overlay.overrideredirect(True)
        self.overlay.attributes("-topmost", True)
        self.overlay.configure(bg=COLORKEY_HEX)

        self.canvas = tk.Canvas(self.overlay, bg=COLORKEY_HEX, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

        # Fit overlay to entire virtual desktop (all monitors, including negative coords)
        self.vl, self.vt, self.vr, self.vb = get_virtual_screen_rect()
        self.vw = self.vr - self.vl
        self.vh = self.vb - self.vt
        self.overlay.geometry(f"{self.vw}x{self.vh}+{self.vl}+{self.vt}")

        # Apply layered window colorkey + make click-through so it never blocks real mouse
        self.overlay.update_idletasks()
        self.hwnd = self.overlay.winfo_id()
        self._apply_colorkey_transparency(self.hwnd)
        self.clickthrough = True
        self._set_clickthrough(self.hwnd, True)

        # --------------- Virtual cursor state ---------------
        # Screen-space (virtual desktop coordinates)
        cx, cy = win32api.GetCursorPos()
        self.cx = clamp(cx, self.vl, self.vr - 1)
        self.cy = clamp(cy, self.vt, self.vb - 1)

        self.cursor_ids = self._create_cursor_gfx()

        # --------------- Interactive targets (demo layer) ---------------
        self.targets = []  # list of dicts: {id, vx, vy, dx, dy, text}
        self._spawn_targets()

        # --------------- Global input (no focus required) ---------------
        # Press SPACE or ENTER to "virtual click"
        self.key_click_space = EdgeKey(win32con.VK_SPACE)
        self.key_click_enter = EdgeKey(win32con.VK_RETURN)

        # Toggle overlay visible with F8 (nice to have)
        self.key_toggle_overlay = EdgeKey(win32con.VK_F8)
        self.overlay_visible = True

        self.last_tick = time.time()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Start tick loop
        self._tick()

    # ---------------- UI ----------------
    def _build_control_ui(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Movement: WASD / Arrow Keys (global) | Shift = faster").pack(anchor="w")
        ttk.Label(frame, text="Virtual click: Space or Enter (global)").pack(anchor="w")
        ttk.Label(frame, text="Toggle overlay: F8").pack(anchor="w", pady=(0, 8))

        row = ttk.Frame(frame)
        row.pack(fill="x", pady=6)
        ttk.Label(row, text="Speed").pack(side="left")
        ttk.Scale(row, from_=1, to=20, variable=self.speed_var, orient="horizontal").pack(side="left", fill="x", expand=True, padx=8)

        row2 = ttk.Frame(frame)
        row2.pack(fill="x", pady=6)
        ttk.Label(row2, text="Targets").pack(side="left")
        ttk.Spinbox(row2, from_=1, to=60, textvariable=self.targets_var, width=5).pack(side="left", padx=8)
        ttk.Button(row2, text="Respawn Targets", command=self._respawn_targets).pack(side="left")

        ttk.Separator(frame).pack(fill="x", pady=8)

        ttk.Label(frame, textvariable=self.pos_var, font=("Consolas", 12)).pack(anchor="w")
        ttk.Label(frame, textvariable=self.score_var).pack(anchor="w", pady=(4, 6))
        ttk.Label(frame, textvariable=self.status).pack(anchor="w")

        btns = ttk.Frame(frame)
        btns.pack(anchor="w", pady=(10, 0))
        ttk.Button(btns, text="Toggle Click-Through", command=self._toggle_clickthrough).pack(side="left")
        ttk.Button(btns, text="Hide/Show Overlay (F8)", command=self._toggle_overlay_visibility).pack(side="left", padx=8)

    # ---------------- Win32 overlay plumbing ----------------
    def _apply_colorkey_transparency(self, hwnd):
        ex = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        ex |= win32con.WS_EX_LAYERED | win32con.WS_EX_TOOLWINDOW
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, ex)

        col = win32api.RGB(*COLORKEY_RGB)
        win32gui.SetLayeredWindowAttributes(hwnd, col, 255, win32con.LWA_COLORKEY)

    def _set_clickthrough(self, hwnd, enabled: bool):
        ex = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        if enabled:
            ex |= win32con.WS_EX_TRANSPARENT
        else:
            ex &= ~win32con.WS_EX_TRANSPARENT
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, ex)

    def _toggle_clickthrough(self):
        self.clickthrough = not self.clickthrough
        self._set_clickthrough(self.hwnd, self.clickthrough)
        self.status.set(f"Overlay click-through is now {'ON' if self.clickthrough else 'OFF'} (still pure / no OS cursor control).")

    def _toggle_overlay_visibility(self):
        self.overlay_visible = not self.overlay_visible
        if self.overlay_visible:
            self.overlay.deiconify()
            self.overlay.lift()
        else:
            self.overlay.withdraw()

    # ---------------- Coordinates ----------------
    def _screen_to_canvas(self, sx, sy):
        return sx - self.vl, sy - self.vt

    # ---------------- Cursor drawing ----------------
    def _create_cursor_gfx(self):
        x, y = self._screen_to_canvas(self.cx, self.cy)
        r = 10
        line = 18
        ids = []
        # ring + crosshair; do NOT use magenta anywhere (it becomes transparent)
        ids.append(self.canvas.create_oval(x - r, y - r, x + r, y + r, outline="lime", width=2, tags=("vcursor",)))
        ids.append(self.canvas.create_line(x - line, y, x + line, y, fill="lime", width=2, tags=("vcursor",)))
        ids.append(self.canvas.create_line(x, y - line, x, y + line, fill="lime", width=2, tags=("vcursor",)))
        ids.append(self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, outline="lime", width=2, tags=("vcursor",)))
        return ids

    def _update_cursor_gfx(self):
        x, y = self._screen_to_canvas(self.cx, self.cy)
        r = 10
        line = 18
        ring, hline, vline, dot = self.cursor_ids
        self.canvas.coords(ring, x - r, y - r, x + r, y + r)
        self.canvas.coords(hline, x - line, y, x + line, y)
        self.canvas.coords(vline, x, y - line, x, y + line)
        self.canvas.coords(dot, x - 2, y - 2, x + 2, y + 2)

        self.pos_var.set(f"x={self.cx:5d}, y={self.cy:5d}   (virtual desktop coords)")

    # ---------------- Targets (demo interactive layer) ----------------
    def _spawn_targets(self):
        self.targets.clear()
        self.canvas.delete("target")

        n = int(self.targets_var.get())
        colors = ["red", "dodgerblue", "green", "orange", "purple", "cyan", "yellow"]

        for i in range(n):
            # random screen-space inside virtual desktop
            sx = random.randint(self.vl + 50, self.vr - 50)
            sy = random.randint(self.vt + 50, self.vb - 50)
            x, y = self._screen_to_canvas(sx, sy)

            text = random.choice(["Alpha", "Beta", "Gamma", "Delta", "Omega", "Kappa", "Zeta", "Nova"])
            cid = self.canvas.create_text(
                x, y,
                text=text,
                font=("Arial", 16, "bold"),
                fill=random.choice(colors),
                tags=("target",)
            )
            dx = random.choice([-1, 1]) * random.randint(2, 5)
            dy = random.choice([-1, 1]) * random.randint(2, 5)

            self.targets.append({"id": cid, "dx": dx, "dy": dy, "text": text})

    def _respawn_targets(self):
        self._spawn_targets()
        self.status.set("Targets respawned.")

    def _move_targets(self):
        # bounce using bbox so text stays visible
        for t in self.targets:
            cid = t["id"]
            bbox = self.canvas.bbox(cid)
            if not bbox:
                continue

            x0, y0, x1, y1 = bbox

            if x0 <= 0:
                t["dx"] = abs(t["dx"])
            elif x1 >= self.vw:
                t["dx"] = -abs(t["dx"])

            if y0 <= 0:
                t["dy"] = abs(t["dy"])
            elif y1 >= self.vh:
                t["dy"] = -abs(t["dy"])

            self.canvas.move(cid, t["dx"], t["dy"])

    # ---------------- Virtual click (pure) ----------------
    def _virtual_click(self):
        # Find what is under the *virtual cursor* in canvas coords
        x, y = self._screen_to_canvas(self.cx, self.cy)
        items = self.canvas.find_overlapping(x, y, x, y)

        # Pick topmost target (ignore cursor gfx)
        hit_id = None
        for cid in reversed(items):
            if "target" in self.canvas.gettags(cid):
                hit_id = cid
                break

        if hit_id is None:
            self.misses += 1
            self.status.set("❌ MISS (virtual click hit nothing).")
        else:
            self.hits += 1
            txt = self.canvas.itemcget(hit_id, "text")
            self.status.set(f"✅ HIT '{txt}' (virtual click).")

            # quick feedback flash
            old = self.canvas.itemcget(hit_id, "fill")
            self.canvas.itemconfig(hit_id, fill="gold")
            self.root.after(120, lambda cid=hit_id, c=old: self._restore_target_color(cid, c))

        self._update_score()

    def _restore_target_color(self, cid, color):
        try:
            self.canvas.itemconfig(cid, fill=color)
        except tk.TclError:
            pass

    def _update_score(self):
        total = self.hits + self.misses
        acc = (self.hits / total * 100.0) if total else 0.0
        self.score_var.set(f"Hits: {self.hits}   Misses: {self.misses}   Acc: {acc:.1f}%")

    # ---------------- Main tick ----------------
    def _tick(self):
        now = time.time()
        dt = now - self.last_tick
        self.last_tick = now

        # Toggle overlay with F8 (edge)
        if self.key_toggle_overlay.pressed():
            self._toggle_overlay_visibility()

        # Movement (global polling)
        base = float(self.speed_var.get())  # 1..20 from slider
        px_per_sec = 110.0 + base * 55.0    # tune feel
        if vk_down(win32con.VK_SHIFT):
            px_per_sec *= 2.2

        step = px_per_sec * dt
        dx = dy = 0.0

        # WASD and arrows
        if vk_down(ord('W')) or vk_down(win32con.VK_UP):
            dy -= step
        if vk_down(ord('S')) or vk_down(win32con.VK_DOWN):
            dy += step
        if vk_down(ord('A')) or vk_down(win32con.VK_LEFT):
            dx -= step
        if vk_down(ord('D')) or vk_down(win32con.VK_RIGHT):
            dx += step

        if dx or dy:
            self.cx = clamp(int(self.cx + dx), self.vl, self.vr - 1)
            self.cy = clamp(int(self.cy + dy), self.vt, self.vb - 1)
            self._update_cursor_gfx()

        # Virtual click (edge)
        if self.key_click_space.pressed() or self.key_click_enter.pressed():
            self._virtual_click()

        # Animate demo targets
        if self.overlay_visible:
            self._move_targets()

        self.root.after(16, self._tick)  # ~60 fps

    def _on_close(self):
        self.root.destroy()

    def run(self):
        self._update_cursor_gfx()
        self.root.mainloop()


if __name__ == "__main__":
    VirtualCursorOverlayApp().run()
