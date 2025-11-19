import os
import cv2

folder_path = "./4/media/lafufu_pictures"
files = os.listdir(folder_path)
files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]

files.sort()  # Deterministic order

scale_factor = 1.0 / 8.0

for filename in files:
    src = os.path.join(folder_path, filename)
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Could not read {src}, skipping.")
        continue

    height, width = img.shape[:2]
    new_size = (int(width * scale_factor), int(height * scale_factor))
    if new_size[0] < 1 or new_size[1] < 1:
        print(f"Image too small after resize: {filename} -> {new_size}")
        continue

    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    # Overwrite original image
    cv2.imwrite(src, resized)
