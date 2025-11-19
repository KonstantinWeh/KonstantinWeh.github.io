import cv2
import numpy as np#

import matplotlib.pyplot as plt

def get_intrinsic_matrix():
    # Create ArUco dictionary and detector parameters (4x4 tags)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # Define tag size in meters (edge length)
    tag_size_m = 0.05
    half = tag_size_m

    # Object points for a single square tag in its own local frame (Z=0 plane)
    # OpenCV ArUco corner ordering is: top-left, top-right, bottom-right, bottom-left (counter-clockwise)
    objp_single_tag = np.array([
        [0.0,        0.0,        0.0],
        [tag_size_m, 0.0,        0.0],
        [tag_size_m, tag_size_m, 0.0],
        [0.0,        tag_size_m, 0.0]
    ], dtype=np.float32)

    object_points_list = []  # list of (N_pts x 3) float32 arrays per image
    image_points_list = []   # list of (N_pts x 2) float32 arrays per image
    image_size = None

    for i in range(34):
        image = cv2.imread(f"./media/aruco_v1/aruco_tag_{i}.JPEG")
        
        image_size = (image.shape[1], image.shape[0])

        corners_list, ids, _ = detector.detectMarkers(image)

        if ids is None or len(corners_list) == 0:
            continue

        # Only use tag ID 0
        ids_flat = ids.flatten()
        tag_id_0_idx = np.where(ids_flat == 0)[0]
        if len(tag_id_0_idx) == 0:
            continue  # Skip if tag ID 0 is not found

        # Use the marker with ID 0
        marker_idx = tag_id_0_idx[0]
        corners = corners_list[marker_idx].reshape(-1, 2).astype(np.float32)

        if corners.shape[0] != 4:
            continue

        object_points_list.append(objp_single_tag.copy())
        image_points_list.append(corners)

    # INSERT_YOUR_CODE
    # Visualize detected corners for each tag on its image (optional)
    corners_images = []
    for idx, (img_idx, corners) in enumerate(zip(range(len(object_points_list)), image_points_list)):
        img = cv2.imread(f"./media/aruco_v1/aruco_tag_{img_idx}.JPEG").copy()
        if img is not None and corners.shape[0] == 4:
            for j, pt in enumerate(corners):
                pt_int = tuple(int(round(x)) for x in pt)
                cv2.circle(img, pt_int, 5, (0, 0, 255), -1)
                cv2.putText(img, str(j), pt_int, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            corners_images.append(img)
            # You could optionally display or save these images, e.g.:
            # cv2.imshow(f"aruco_tag_{img_idx}", img)
            # cv2.waitKey(200) # pause a bit
            # Or save with: cv2.imwrite(f"./media/aruco_v1/aruco_tag_{img_idx}_corners.jpeg", img)

    

    # Calibrate if we have enough views
    if len(object_points_list) >= 3 and image_size is not None:
        rms, camera_matrix, dist_coeffs, rvecs, svecs = cv2.calibrateCamera(
            objectPoints=object_points_list,
            imagePoints=image_points_list,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None
        )

        print("Calibration RMS reprojection error:", rms)
        print("Camera matrix (K):\n", camera_matrix)
        print("Distortion coefficients (k1,k2,p1,p2,k3,...):\n", dist_coeffs.ravel())
        return camera_matrix, dist_coeffs
    else:
        print("Not enough detections for calibration. Need at least 3 images with a detected tag.")
        return None, None


if __name__ == "__main__":
    camera_matrix, dist_coeffs = get_intrinsic_matrix()
    print(camera_matrix)
    print(dist_coeffs)