import tkinter as tk
import random
import csv

# ---------------- Window Setup ----------------
root = tk.Tk()
root.title("Text Clicker")
root.geometry("800x500")
root.resizable(True, True)

canvas = tk.Canvas(root, bg="white")
canvas.pack(fill=tk.BOTH, expand=True)

# ---------------- Words ----------------
WORDS = [
    ("Apple", "Fruit"),
    ("Banana", "Fruit"),
    ("Cat", "Animal"),
    ("Dog", "Animal"),
    ("Carrot", "Vegetable"),
    ("Elephant", "Animal"),
    ("Orange", "Fruit"),
    ("Broccoli", "Vegetable")
]

NUM_WORDS = 6
active_words = []  # [text, category, x, y, dx, dy, canvas_id]
click_positions = []  # (word, category, hit)
moving = False

# ---------------- Functions ----------------
def spawn_words():
    """Spawn moving words at random positions without overlap"""
    global active_words
    canvas.delete("word")
    active_words = []

    canvas.update_idletasks()  # ensure correct canvas size
    canvas_width = max(canvas.winfo_width(), 200)
    canvas_height = max(canvas.winfo_height(), 100)

    positions = []

    for _ in range(NUM_WORDS):
        text, category = random.choice(WORDS)

        # Avoid overlapping positions
        for _ in range(100):  # max attempts
            x = random.randint(50, canvas_width - 50)
            y = random.randint(50, canvas_height - 50)
            if all(abs(x - px) > 50 and abs(y - py) > 30 for px, py in positions):
                positions.append((x, y))
                break

        dx, dy = random.choice([-3, 3]), random.choice([-3, 3])
        cid = canvas.create_text(
            x, y, text=text, font=("Arial", 16, "bold"),
            fill=random.choice(["red", "blue", "green", "purple"]),
            tags="word"
        )

        active_words.append([text, category, x, y, dx, dy, cid])

def move_words():
    """Animate words around the canvas"""
    if not moving:
        return

    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()

    for word in active_words:
        text, category, x, y, dx, dy, cid = word

        x += dx
        y += dy

        # Simple bounding box for bouncing (approx 50x20)
        if x <= 0 or x >= canvas_width:
            dx *= -1
        if y <= 0 or y >= canvas_height:
            dy *= -1

        word[2], word[3], word[4], word[5] = x, y, dx, dy
        canvas.coords(cid, x, y)

    root.after(30, move_words)

def toggle_movement():
    global moving
    moving = not moving
    if moving:
        move_button.config(text="Stop Moving")
        move_words()
    else:
        move_button.config(text="Start Moving")

def on_click(event):
    """Check if a word was clicked"""
    hit_any = False
    for word in active_words:
        text, category, x, y, dx, dy, cid = word
        bbox = canvas.bbox(cid)  # bounding box of text
        if bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]:
            print(f"Click at ({event.x}, {event.y}) → ✅ HIT '{text}' ({category})")
            click_positions.append((text, category, True))
            hit_any = True
            break
    if not hit_any:
        print(f"Click at ({event.x}, {event.y}) → ❌ MISS")
        click_positions.append(("", "", False))

def on_close():
    """Save click data to CSV"""
    with open("text_click_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Word", "Category", "Hit"])
        for row in click_positions:
            writer.writerow(row)
    print("Data saved to text_click_data.csv")
    root.destroy()

# ---------------- UI Bindings ----------------
canvas.bind("<Button-1>", on_click)
move_button = tk.Button(root, text="Start Moving", command=toggle_movement)
move_button.pack()

spawn_words()
root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
