import tkinter as tk
import random
import csv

# Window setup
root = tk.Tk()
root.title("Apple Clicker")
root.geometry("600x400")
root.resizable(True, True)  # Make window resizable

# Create canvas with initial size
canvas = tk.Canvas(root, bg="white")
canvas.pack(fill=tk.BOTH, expand=True)  # Canvas fills window

# Store history
apple_positions = []   # stores (x, y, left, right, top, bottom)
click_positions = []   # stores (x, y, hit/miss)

APPLE_SIZE = 30
current_bounds = None  # keep track of active apple bounds

def spawn_apple():
    """Spawn apple at random position on canvas"""
    global current_bounds
    canvas.delete("apple")

    # Get current canvas dimensions
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()

    # Ensure minimum dimensions to prevent negative values
    canvas_width = max(canvas_width, APPLE_SIZE)
    canvas_height = max(canvas_height, APPLE_SIZE)

    x = random.randint(0, canvas_width - APPLE_SIZE)
    y = random.randint(0, canvas_height - APPLE_SIZE)

    left = x
    right = x + APPLE_SIZE
    top = y
    bottom = y + APPLE_SIZE

    current_bounds = (left, right, top, bottom)
    apple_positions.append((x, y, left, right, top, bottom))

    # Draw apple as red circle
    canvas.create_oval(left, top, right, bottom,
                      fill="red", outline="black", tags="apple")

    print(f"Apple spawned → Left: {left}, Right: {right}, Top: {top}, Bottom: {bottom}")

def on_click(event):
    """Record clicks, check if apple clicked, respawn apple"""
    global current_bounds
    left, right, top, bottom = current_bounds
    hit = left <= event.x <= right and top <= event.y <= bottom
    click_positions.append((event.x, event.y, hit))

    if hit:
        print(f"Click at ({event.x}, {event.y}) → ✅ HIT the apple!")
    else:
        print(f"Click at ({event.x}, {event.y}) → ❌ MISS")

    spawn_apple()

def toggle_fullscreen(event=None):
    """Toggle between fullscreen and windowed mode"""
    root.attributes("-fullscreen", not root.attributes("-fullscreen"))
    if not root.attributes("-fullscreen"):
        root.geometry("600x400")  # Restore default size when exiting fullscreen
    spawn_apple()  # Respawn apple to ensure it fits new window size

# Bind click and fullscreen toggle
canvas.bind("<Button-1>", on_click)
root.bind("<F>", toggle_fullscreen)

# Handle window resize
def on_resize(event):
    """Respawn apple when window is resized"""
    spawn_apple()

canvas.bind("<Configure>", on_resize)

# First apple
spawn_apple()

def on_close():
    """Save all data to CSV on close"""
    with open("apple_click_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Type", "X", "Y", "Left", "Right", "Top", "Bottom", "Hit"])
        for (x, y, left, right, top, bottom) in apple_positions:
            writer.writerow(["Apple", x, y, left, right, top, bottom, ""])
        for (x, y, hit) in click_positions:
            writer.writerow(["Click", x, y, "", "", "", "", hit])

    print("Data saved to apple_click_data.csv")
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()