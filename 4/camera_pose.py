import glob
import json
import time
from pathlib import Path
from calibration import get_intrinsic_matrix
import matplotlib.pyplot as plt

import cv2
import numpy as np
import viser

if __name__ == "__main__":
    IMG_DIR = "./media/football/*.JPEG"
    TAG_SIZE_M = 0.05

    K, distCoeffs = get_intrinsic_matrix()
    if K is None or distCoeffs is None:
        print("ERROR: Could not get camera intrinsics. Calibration may have failed.")
        exit(1)

    ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    def make_square_tag_object_points(size_m):
        s = size_m
        pts = np.array([
            [0.0, 0.0, 0.0],
            [s,   0.0, 0.0],
            [s,   s,   0.0],
            [0.0, s,   0.0],
        ], dtype=np.float64)
        return pts.reshape(-1, 1, 3)

    objectPoints = make_square_tag_object_points(TAG_SIZE_M)

    img_paths = sorted(glob.glob(IMG_DIR))

    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())
    poses = []
    failed = []

    vis_images = []
    for i, p in enumerate(img_paths):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[skip] Could not read {p}")
            failed.append(p)
            continue

        H, W = img.shape[:2]
        corners_list, ids, _ = detector.detectMarkers(img)

        if ids is None or len(ids) == 0 or len(corners_list) == 0:
            print(f"[skip] No ArUco detected in {Path(p).name}")
            failed.append(p)
            continue

        # Use the first detected marker (or you could filter by specific ID)
        # Validate that the marker has exactly 4 corners
        corners_flat = corners_list[0].reshape(-1, 2)
        if corners_flat.shape[0] != 4:
            print(f"[skip] Invalid marker corners in {Path(p).name}")
            failed.append(p)
            continue

        imagePoints = corners_list[0].reshape(-1, 1, 2).astype(np.float64)

        pnp_flag = cv2.SOLVEPNP_IPPE_SQUARE if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE") else cv2.SOLVEPNP_ITERATIVE

        # Show the image with detected corners overlaid
        img_vis = img.copy()
        cv2.polylines(
            img_vis,
            [corners_flat.astype(np.int32).reshape(-1, 1, 2)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2
        )
        for idx, pt in enumerate(corners_flat):
            pt = tuple(int(x) for x in pt)
            cv2.circle(img_vis, pt, 6, (0, 0, 255), -1)
            cv2.putText(img_vis, str(idx), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        vis_images.append(img_vis.copy())

        success, rvec, tvec = cv2.solvePnP(
            objectPoints,
            imagePoints,
            K,
            distCoeffs=None,
            flags=pnp_flag
        )

        if not success:
            print(f"[skip] solvePnP failed for {Path(p).name}")
            failed.append(p)
            continue

        R_wc, _ = cv2.Rodrigues(rvec)
        t_wc = tvec.reshape(3, 1)

        w2c = np.hstack([R_wc, t_wc])

        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc
        c2w = np.hstack([R_cw, t_cw])

        poses.append({
            "index": i,
            "path": p,
            "success": True,
            "w2c": w2c.tolist(),
            "c2w": c2w.tolist(),
            "H": int(H),
            "W": int(W)
        })


    print(f"\nDone. Good: {len(poses)} | Skipped: {len(failed)}")
    if failed:
        print("Skipped files:", [Path(x).name for x in failed])

    out_json = "poses_aruco.json"
    with open(out_json, "w") as f:
        json.dump(poses, f, indent=2)
    print(f"Saved: {out_json}")

    print("1")
    if not poses:
        print("ERROR: No valid poses found. Cannot visualize.")
        exit(1)
    
    server = viser.ViserServer(share=False)

    for cam in poses:
        p = cam["path"]
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        c2w = np.array(cam["c2w"], dtype=np.float64)
        H, W = cam["H"], cam["W"]

        fx, fy = K[0, 0], K[1, 1]
        v_fov = 2.0 * np.arctan2(H / 2.0, fy)
        aspect = W / H

        R = c2w[:, :3]
        t = c2w[:, 3]
        quat_wxyz = viser.transforms.SO3.from_matrix(R).wxyz

        server.scene.add_camera_frustum(
            f"/cameras/{cam['index']}",
            fov=v_fov,
            aspect=aspect,
            scale=0.02,
            wxyz=quat_wxyz,
            position=t,
            image=img_rgb
        )
    
    print("Viser is live. Open the share URL printed above.")
    while True:
        time.sleep(0.1)
