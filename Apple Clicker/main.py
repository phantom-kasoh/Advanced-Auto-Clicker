import tkinter as tk
from tkinter import ttk
import random
import csv
import time
from datetime import datetime


class TextClickerApp:
    WORDS = [
        ("Apple", "Fruit"),
        ("Banana", "Fruit"),
        ("Cat", "Animal"),
        ("Dog", "Animal"),
        ("Carrot", "Vegetable"),
        ("Elephant", "Animal"),
        ("Orange", "Fruit"),
        ("Broccoli", "Vegetable"),
    ]

    COLORS = ["red", "blue", "green", "purple", "darkorange", "teal"]
    FONT = ("Arial", 16, "bold")

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Text Clicker")
        self.root.geometry("800x500")
        self.root.resizable(True, True)

        # ---- State ----
        self.moving = False
        self.after_id = None
        self.tick_ms = 30

        # cid -> dict(text/category/dirx/diry/color/shape)
        self.word_items = {}
        self.placed_bboxes = []  # list of (x0,y0,x1,y1) to avoid overlaps

        self.click_log = []  # list of dict rows for CSV
        self.hits = 0
        self.misses = 0

        # mode: "words" or "dots"
        self.mode_var = tk.StringVar(value="words")

        # ---- UI ----
        self._build_ui()

        # initial content
        self.spawn_words()

        # close hook
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)

        self.start_stop_btn = ttk.Button(top, text="Start Moving", command=self.toggle_movement)
        self.start_stop_btn.pack(side=tk.LEFT)

        self.respawn_btn = ttk.Button(top, text="Respawn Words", command=self.spawn_words)
        self.respawn_btn.pack(side=tk.LEFT, padx=(8, 0))

        ttk.Button(top, text="Reset Stats", command=self.reset_stats).pack(side=tk.LEFT, padx=(8, 0))

        # controls
        controls = ttk.Frame(top)
        controls.pack(side=tk.LEFT, padx=16)

        ttk.Label(controls, text="Speed").grid(row=0, column=0, sticky="w")
        self.speed_var = tk.IntVar(value=3)
        self.speed_slider = ttk.Scale(
            controls, from_=1, to=10, orient="horizontal",
            command=lambda _v: None
        )
        self.speed_slider.set(self.speed_var.get())
        self.speed_slider.grid(row=0, column=1, padx=8, sticky="we")
        controls.columnconfigure(1, weight=1)

        def sync_speed(_event=None):
            self.speed_var.set(int(round(self.speed_slider.get())))

        self.speed_slider.bind("<ButtonRelease-1>", sync_speed)
        self.speed_slider.bind("<B1-Motion>", sync_speed)

        ttk.Label(controls, text="Count").grid(row=0, column=2, padx=(12, 0), sticky="w")
        self.num_words_var = tk.IntVar(value=6)
        self.num_words_spin = ttk.Spinbox(
            controls, from_=1, to=30, textvariable=self.num_words_var, width=4
        )
        self.num_words_spin.grid(row=0, column=3, padx=(8, 0), sticky="w")

        # mode toggle (Words vs Dots)
        mode_frame = ttk.Frame(top)
        mode_frame.pack(side=tk.LEFT, padx=16)

        ttk.Label(mode_frame, text="Mode").pack(side=tk.LEFT)
        ttk.Radiobutton(
            mode_frame, text="Words", value="words",
            variable=self.mode_var, command=self._on_mode_change
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Radiobutton(
            mode_frame, text="Dots", value="dots",
            variable=self.mode_var, command=self._on_mode_change
        ).pack(side=tk.LEFT, padx=(8, 0))

        # scoreboard
        self.score_var = tk.StringVar()
        self._update_score_text()
        ttk.Label(top, textvariable=self.score_var).pack(side=tk.RIGHT)

        # canvas
        self.canvas = tk.Canvas(self.root, bg="white", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # bindings
        self.canvas.bind("<Button-1>", self.on_click)

    def _on_mode_change(self):
        # update respawn button text and respawn items
        self.respawn_btn.config(text="Respawn Dots" if self.mode_var.get() == "dots" else "Respawn Words")
        self.spawn_words()

    # ---------------- Core ----------------
    def spawn_words(self):
        """Spawn items at random non-overlapping positions (words OR dots depending on mode)."""
        self.canvas.delete("word")
        self.word_items.clear()
        self.placed_bboxes.clear()

        self.canvas.update_idletasks()
        w = max(self.canvas.winfo_width(), 200)
        h = max(self.canvas.winfo_height(), 120)

        n = int(self.num_words_var.get())

        mode = self.mode_var.get()
        if mode == "words":
            items = self._pick_words(n)  # list of (text, category)
        else:
            # dots: text/category will be synthesized
            items = [("Dot", random.choice(self.COLORS)) for _ in range(n)]

        # placement params
        padding = 10
        margin = 20
        dot_r = 9  # radius for dots

        for (text, category) in items:
            cid = None
            last_bbox = None

            for _attempt in range(300):
                x = random.randint(margin, max(margin, w - margin))
                y = random.randint(margin, max(margin, h - margin))

                if mode == "words":
                    color = random.choice(self.COLORS)
                    cid = self.canvas.create_text(
                        x, y, text=text, font=self.FONT, fill=color, tags=("word",)
                    )
                else:
                    # category is the color in dot mode
                    color = category
                    cid = self.canvas.create_oval(
                        x - dot_r, y - dot_r, x + dot_r, y + dot_r,
                        fill=color, outline="", tags=("word",)
                    )

                bbox = self.canvas.bbox(cid)  # (x0,y0,x1,y1)
                if not bbox:
                    self.canvas.delete(cid)
                    cid = None
                    continue

                x0, y0, x1, y1 = bbox

                # Keep fully inside canvas
                if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                    self.canvas.delete(cid)
                    cid = None
                    continue

                # Overlap check (bbox with padding)
                padded = (x0 - padding, y0 - padding, x1 + padding, y1 + padding)
                if any(self._rects_overlap(padded, other) for other in self.placed_bboxes):
                    self.canvas.delete(cid)
                    cid = None
                    continue

                last_bbox = padded
                break

            # Fallback if placement failed
            if cid is None:
                x = random.randint(margin, max(margin, w - margin))
                y = random.randint(margin, max(margin, h - margin))

                if mode == "words":
                    color = random.choice(self.COLORS)
                    cid = self.canvas.create_text(
                        x, y, text=text, font=self.FONT, fill=color, tags=("word",)
                    )
                else:
                    color = category
                    cid = self.canvas.create_oval(
                        x - dot_r, y - dot_r, x + dot_r, y + dot_r,
                        fill=color, outline="", tags=("word",)
                    )

                bbox = self.canvas.bbox(cid) or (x, y, x + 1, y + 1)
                x0, y0, x1, y1 = bbox
                last_bbox = (x0 - padding, y0 - padding, x1 + padding, y1 + padding)

            self.placed_bboxes.append(last_bbox)

            # Store metadata (keep "word/category" fields for CSV compatibility)
            if mode == "words":
                stored_text = text
                stored_cat = category
                shape = "text"
            else:
                stored_text = "Dot"
                stored_cat = color  # store dot color in "category"
                shape = "dot"

            self.word_items[cid] = {
                "text": stored_text,
                "category": stored_cat,
                "dirx": random.choice([-1, 1]),
                "diry": random.choice([-1, 1]),
                "color": color,
                "shape": shape,
            }

    def move_words(self):
        if not self.moving:
            self.after_id = None
            return

        self.canvas.update_idletasks()
        w = max(self.canvas.winfo_width(), 1)
        h = max(self.canvas.winfo_height(), 1)

        speed = max(1, int(self.speed_var.get()))

        for cid, info in list(self.word_items.items()):
            bbox = self.canvas.bbox(cid)
            if not bbox:
                continue

            x0, y0, x1, y1 = bbox

            # Bounce based on actual bbox, not the center point.
            if x0 <= 0:
                info["dirx"] = 1
            elif x1 >= w:
                info["dirx"] = -1

            if y0 <= 0:
                info["diry"] = 1
            elif y1 >= h:
                info["diry"] = -1

            dx = info["dirx"] * speed
            dy = info["diry"] * speed
            self.canvas.move(cid, dx, dy)

        self.after_id = self.root.after(self.tick_ms, self.move_words)

    def toggle_movement(self):
        self.moving = not self.moving
        self.start_stop_btn.config(text="Stop Moving" if self.moving else "Start Moving")

        if self.moving and self.after_id is None:
            self.move_words()

    # ---------------- Interaction + Logging ----------------
    def on_click(self, event: tk.Event):
        items = self.canvas.find_overlapping(event.x, event.y, event.x, event.y)
        hit_cid = None
        for cid in reversed(items):
            if cid in self.word_items:
                hit_cid = cid
                break

        t = time.time()
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        if hit_cid is not None:
            info = self.word_items[hit_cid]
            coords = self.canvas.coords(hit_cid)
            bbox = self.canvas.bbox(hit_cid)

            self.hits += 1
            self._update_score_text()

            # flash fill (works for text and dots)
            original = info["color"]
            try:
                self.canvas.itemconfig(hit_cid, fill="gold")
            except tk.TclError:
                pass
            self.root.after(120, lambda cid=hit_cid, col=original: self._restore_color(cid, col))

            # For ovals, coords is [x0,y0,x1,y1]; for text, coords is [x,y]
            if coords and len(coords) == 2:
                cx, cy = coords
            elif coords and len(coords) == 4:
                cx = (coords[0] + coords[2]) / 2
                cy = (coords[1] + coords[3]) / 2
            else:
                cx = cy = ""

            self.click_log.append({
                "timestamp_unix": t,
                "click_x": event.x,
                "click_y": event.y,
                "moving": int(self.moving),
                "hit": 1,
                "word": info["text"],
                "category": info["category"],
                "canvas_item_id": hit_cid,
                "word_center_x": cx,
                "word_center_y": cy,
                "word_bbox": ",".join(map(str, bbox)) if bbox else "",
                "canvas_w": w,
                "canvas_h": h,
                "speed": int(self.speed_var.get()),
            })
        else:
            self.misses += 1
            self._update_score_text()

            self.click_log.append({
                "timestamp_unix": t,
                "click_x": event.x,
                "click_y": event.y,
                "moving": int(self.moving),
                "hit": 0,
                "word": "",
                "category": "",
                "canvas_item_id": "",
                "word_center_x": "",
                "word_center_y": "",
                "word_bbox": "",
                "canvas_w": w,
                "canvas_h": h,
                "speed": int(self.speed_var.get()),
            })

    def _restore_color(self, cid: int, color: str):
        if cid in self.word_items:
            try:
                self.canvas.itemconfig(cid, fill=color)
            except tk.TclError:
                pass

    def reset_stats(self):
        self.hits = 0
        self.misses = 0
        self.click_log.clear()
        self._update_score_text()

    def on_close(self):
        self.moving = False
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

        fname = datetime.now().strftime("text_click_data_%Y%m%d_%H%M%S.csv")
        self._save_csv(fname)
        print(f"Saved: {fname}")

        self.root.destroy()

    def _save_csv(self, filename: str):
        fieldnames = [
            "timestamp_unix",
            "click_x", "click_y",
            "moving",
            "hit",
            "word", "category",
            "canvas_item_id",
            "word_center_x", "word_center_y",
            "word_bbox",
            "canvas_w", "canvas_h",
            "speed",
        ]

        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.click_log:
                writer.writerow(row)

    # ---------------- Helpers ----------------
    def _pick_words(self, n: int):
        if n <= len(self.WORDS):
            return random.sample(self.WORDS, k=n)
        out = list(self.WORDS)
        while len(out) < n:
            out.append(random.choice(self.WORDS))
        random.shuffle(out)
        return out

    @staticmethod
    def _rects_overlap(a, b) -> bool:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        return not (ax1 < bx0 or ax0 > bx1 or ay1 < by0 or ay0 > by1)

    def _update_score_text(self):
        total = self.hits + self.misses
        acc = (self.hits / total * 100.0) if total else 0.0
        self.score_var.set(f"Hits: {self.hits}   Misses: {self.misses}   Accuracy: {acc:.1f}%   Clicks: {total}")


if __name__ == "__main__":
    root = tk.Tk()
    app = TextClickerApp(root)
    root.mainloop()
