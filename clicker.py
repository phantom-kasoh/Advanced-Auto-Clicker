import tkinter as tk
import win32api
import time
import threading
import requests
import sys

stop_program = False
VK_BACKTICK = 0xC0

def mouseCords():
    # Create the main top-level window
    root = tk.Tk()
    root.overrideredirect(True)  # remove window decorations
    root.attributes("-topmost", True)  # always on top
    root.config(bg="black")

    # Label to display coordinates
    label = tk.Label(root, text="", fg="white", bg="black", font=("Arial", 12))
    label.pack(padx=5, pady=5)


    # Function to update mouse position
    def update_position():
        x, y = win32api.GetCursorPos()  # get global mouse coordinates
        label.config(text=f"X: {x}  Y: {y}")

        # Smoothly move the box towards the cursor
        current_x = root.winfo_x()
        current_y = root.winfo_y()

        # calculate small step toward target
        step_x = int((x + 20 - current_x) * 0.3)
        step_y = int((y + 20 - current_y) * 0.3)

        new_x = current_x + step_x
        new_y = current_y + step_y

        root.geometry(f"+{new_x}+{new_y}")
        root.after(10, update_position)  # repeat every 10ms for smooth movement


    # Start updating
    update_position()
    root.mainloop()

### new code in here ###
def get_api_data():
    r = requests.get("http://127.0.0.1:5000/apple_coords")
    data = r.json()
    return data

def move(x, y):
    win32api.SetCursorPos((x, y))

def move_to_red_circle():
    start_time = time.time()
    elt = 0
    while elt < 30:
        data = get_api_data()
        move(data['center_x'], data['center_y'])
        end_time = time.time()
        elt = end_time - start_time

##########################


thread1 = threading.Thread(target=mouseCords)
#thread2 = threading.Thread(target=move_to_red_circle)


thread1.start()
#thread2.start()


thread1.join()
#thread2.join()
