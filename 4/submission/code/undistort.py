import json
from pathlib import Path
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from calibration import get_intrinsic_matrix


POSES_JSON = "poses_aruco.json"
OUT_NPZ = "lafufu_data.npz"

VAL_FRAC = 0.10
TEST_FRAC = 0.10
RANDOM_SEED = 42

K, distCoeffs = get_intrinsic_matrix()

def to_homogeneous_4x4(c2w_3x4):
    M = np.eye(4, dtype=np.float64)
    M[:3, :4] = c2w_3x4
    return M

with open(POSES_JSON, "r") as f:
    pose_entries = json.load(f)

valid = []
for e in pose_entries:
    p = Path(e["path"])
    if not p.exists():
        print(f"[skip] Missing image: {p}")
        continue
    c2w = np.array(e["c2w"], dtype=np.float64)
    valid.append((p, c2w))

imgs = []
c2ws_4x4 = []

for idx, (img_path, c2w_3x4) in enumerate(valid):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

    if img is None:
        print(f"[skip] Could not read {img_path}")
        continue

    # plt.imshow(img)
    # plt.show()

    und = cv2.undistort(img, K, distCoeffs)

    und = cv2.cvtColor(und, cv2.COLOR_BGR2RGB)

    # plt.imshow(und)
    # plt.show()
    
    imgs.append(und)
    c2ws_4x4.append(to_homogeneous_4x4(c2w_3x4))

H, W = imgs[0].shape[:2]
for i, im in enumerate(imgs):
    if im.shape[:2] != (H, W):
        raise ValueError(f"Image {i} has size {im.shape[:2]} != {(H, W)} after processing.")

images = np.stack(imgs, axis=0).astype(np.uint8)
c2ws = np.stack(c2ws_4x4, axis=0).astype(np.float32)

N = images.shape[0]
print(f"Prepared {N} images at {H}x{W}")

idx_all = np.arange(N)
idx_train, idx_tmp = train_test_split(
    idx_all, test_size=VAL_FRAC + TEST_FRAC, random_state=RANDOM_SEED, shuffle=True)
rel_test = TEST_FRAC / (VAL_FRAC + TEST_FRAC) if (VAL_FRAC + TEST_FRAC) > 0 else 0.0
idx_val, idx_test = train_test_split(
    idx_tmp, test_size=rel_test, random_state=RANDOM_SEED, shuffle=True)

def take(arr, ids):
    return arr[ids] if len(ids) > 0 else arr[:0]

images_train = take(images, idx_train)
images_val   = take(images, idx_val)
c2ws_train   = take(c2ws, idx_train)
c2ws_val     = take(c2ws, idx_val)
c2ws_test    = take(c2ws, idx_test)

fx, fy = K[0, 0], K[1, 1]
focal = float(0.5 * (fx + fy))

print(f"fx={fx:.3f}, fy={fy:.3f}, focal={focal:.3f}")

np.savez(
    OUT_NPZ,
    images_train=images_train,
    c2ws_train=c2ws_train,
    images_val=images_val,
    c2ws_val=c2ws_val,
    c2ws_test=c2ws_test,
    focal=focal
)
print(f"Saved dataset: {OUT_NPZ}")
print(f"Split sizes: train={len(images_train)}, val={len(images_val)}, test={len(c2ws_test)}")
