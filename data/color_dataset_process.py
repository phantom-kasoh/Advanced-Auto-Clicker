import os
import cv2
from sklearn.model_selection import train_test_split

# ---------- CONFIG ----------
INPUT_DIR = r"C:\Users\phant\Downloads\colors_dataset"  # folder containing black, blue, brown, etc.
OUTPUT_DIR = r"C:\Users\phant\PycharmProjects\Advanced_autoclicker\advanced autoclicker\data"
IMG_SIZE = (64, 64)
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------- UTILITY FUNCTION ----------
def process_and_copy(images_list, labels_list, dest_dir):
    for img_path, label in zip(images_list, labels_list):
        label_dir = os.path.join(dest_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_resized = cv2.resize(img, IMG_SIZE)
        cv2.imwrite(os.path.join(label_dir, img_name), img_resized)

# ---------- LOAD ALL IMAGES ----------
all_images = []
all_labels = []

for color_class in os.listdir(INPUT_DIR):
    color_dir = os.path.join(INPUT_DIR, color_class)
    if os.path.isdir(color_dir):
        for img_name in os.listdir(color_dir):
            if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                all_images.append(os.path.join(color_dir, img_name))
                all_labels.append(color_class)

# ---------- SPLIT TRAIN/TEST ----------
train_imgs, test_imgs, train_labels, test_labels = train_test_split(
    all_images, all_labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
)

# ---------- COPY AND PROCESS ----------
process_and_copy(train_imgs, train_labels, os.path.join(OUTPUT_DIR, "train/colors"))
process_and_copy(test_imgs, test_labels, os.path.join(OUTPUT_DIR, "test/colors"))

print(f"Processed {len(train_imgs)} training images and {len(test_imgs)} test images for colors.")
