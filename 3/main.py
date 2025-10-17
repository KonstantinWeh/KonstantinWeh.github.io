import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr
import cv2
from PIL import Image




def get_points(im1, im2, n=4):
    print('Please select ', n, ' points in each image for alignment.')

    im1_pts = np.zeros((n, 2))
    im2_pts = np.zeros((n, 2))

    for i in range(n):
        plt.imshow(im1)
        p1 = plt.ginput(1)
        plt.close()
        plt.imshow(im2)
        p2 = plt.ginput(1)
        plt.close()
        im1_pts[i] = [p1[0][0], p1[0][1]]
        im2_pts[i] = [p2[0][0], p2[0][1]]

    return im1_pts, im2_pts

def computeH(im1_pts, im2_pts):

    im1_pts = np.asarray(im1_pts, dtype=float)
    im2_pts = np.asarray(im2_pts, dtype=float)
    if im1_pts.shape != im2_pts.shape or im1_pts.shape[1] != 2:
        raise ValueError("im1_pts and im2_pts must both be (n,2).")
    n = im1_pts.shape[0]
    if n < 4:
        raise ValueError("Need at least 4 point correspondences.")

    x, y = im1_pts[:, 0], im1_pts[:, 1]
    u, v = im2_pts[:, 0], im2_pts[:, 1]

    A = np.zeros((2 * n, 8), dtype=float)
    b = np.zeros(2 * n, dtype=float)

    A[0::2, 0:3] = np.stack([x, y, np.ones(n)], axis=1)
    A[1::2, 3:6] = np.stack([x, y, np.ones(n)], axis=1)
    A[0::2, 6]   = -u * x
    A[0::2, 7]   = -u * y
    A[1::2, 6]   = -v * x
    A[1::2, 7]   = -v * y

    b[0::2] = u
    b[1::2] = v

    h, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0]
    ])

    return H


import numpy as np

def warp_image_nearest_neighbor(im, H, stiching_on_right=False):
    if H.shape != (3, 3):
        raise ValueError("H must be 3x3.")

    h_in, w_in = im.shape[:2]
    # You can predict the bounding box by piping the four corners of the image through your H transform.
    corners = np.array([[0, 0, 1], [w_in, 0, 1], [w_in, h_in, 1], [0, h_in, 1]])

    corners_transformed = H @ corners.T
    corners_transformed = corners_transformed.T
    
    # normalize
    corners_transformed = corners_transformed[:, :2] / corners_transformed[:, 2:3]
    
    x_min = int(np.floor(np.min(corners_transformed[:, 0])))
    y_min = int(np.floor(np.min(corners_transformed[:, 1])))
    x_max = int(np.ceil(np.max(corners_transformed[:, 0])))
    y_max = int(np.ceil(np.max(corners_transformed[:, 1])))

#   if stiching on the right side, use this 
    if stiching_on_right:
        w_out = x_max
        h_out = y_max
    else:
        w_out = x_max - x_min
        h_out = y_max - y_min
    
    
    translation = np.array([[1, 0, -min(x_min, 0)], [0, 1, -min(y_min, 0)], [0, 0, 1]])
    H_adjusted = translation @ H

    Hinv = np.linalg.inv(H_adjusted)

    u = np.arange(w_out)
    v = np.arange(h_out)
    U, V = np.meshgrid(u, v)

    ones = np.ones_like(U, dtype=float)
    ref_h = np.stack([U.ravel(), V.ravel(), ones.ravel()], axis=0)  

    src_h = Hinv @ ref_h
    w = src_h[2, :]
    # no divide by zero
    valid = (w != 0) & np.isfinite(w)

    x = np.empty_like(w, dtype=float); x.fill(np.nan)
    y = np.empty_like(w, dtype=float); y.fill(np.nan)

    # normalize
    x[valid] = src_h[0, valid] / w[valid]
    y[valid] = src_h[1, valid] / w[valid]
    
    # Nearest Neighbor Interpolation: Round coordinates to the nearest pixel value
    xi = np.rint(x).astype(np.int64)
    yi = np.rint(y).astype(np.int64)

    # Pay attention to how you treat pixels which don't have any values.
    h_in, w_in = im.shape[:2]
    inbounds = valid & (xi >= 0) & (xi < w_in) & (yi >= 0) & (yi < h_in)

    imwarped = np.zeros((h_out, w_out) + ((im.shape[2],)), dtype=im.dtype)
    flat_out = imwarped.reshape(-1, im.shape[2])
    flat_out[inbounds, :] = im[yi[inbounds], xi[inbounds], :]
    imwarped_nn = flat_out.reshape(h_out, w_out, im.shape[2])

    return imwarped_nn

def warp_image_bilinear(im, H, stiching_on_right=False):
    if H.shape != (3, 3):
        raise ValueError("H must be 3x3.")

    h_in, w_in = im.shape[:2]
    # You can predict the bounding box by piping the four corners of the image through your H transform.
    corners = np.array([[0, 0, 1], [w_in, 0, 1], [w_in, h_in, 1], [0, h_in, 1]])

    corners_transformed = H @ corners.T
    corners_transformed = corners_transformed.T
    
    # normalize
    corners_transformed = corners_transformed[:, :2] / corners_transformed[:, 2:3]
    
    x_min = int(np.floor(np.min(corners_transformed[:, 0])))
    y_min = int(np.floor(np.min(corners_transformed[:, 1])))
    x_max = int(np.ceil(np.max(corners_transformed[:, 0])))
    y_max = int(np.ceil(np.max(corners_transformed[:, 1])))


    if stiching_on_right:
        w_out = x_max
        h_out = y_max
    else:
        w_out = x_max - x_min
        h_out = y_max - y_min

    translation = np.array([[1, 0, -min(x_min, 0)], [0, 1, -min(y_min, 0)], [0, 0, 1]])
    H_adjusted = translation @ H

    Hinv = np.linalg.inv(H_adjusted)

    u = np.arange(w_out)
    v = np.arange(h_out)
    U, V = np.meshgrid(u, v)

    ones = np.ones_like(U, dtype=float)
    ref_h = np.stack([U.ravel(), V.ravel(), ones.ravel()], axis=0)  

    src_h = Hinv @ ref_h
    w = src_h[2, :]
    # no divide by zero
    valid = (w != 0) & np.isfinite(w)

    x = np.empty_like(w, dtype=float); x.fill(np.nan)
    y = np.empty_like(w, dtype=float); y.fill(np.nan)

    # normalize
    x[valid] = src_h[0, valid] / w[valid]
    y[valid] = src_h[1, valid] / w[valid]

    # Bilinear Interpolation needs to take care of the boundary cases
    h_in, w_in = im.shape[:2]
    inbounds = valid & (x >= 0.0) & (x <= (w_in - 1)) & (y >= 0.0) & (y <= (h_in - 1))

    # Bilinear Interpolation: Use weighted average of four neighboring pixels
    x0 = np.floor(x[inbounds]).astype(np.int64)
    y0 = np.floor(y[inbounds]).astype(np.int64)
    x1 = np.minimum(x0 + 1, w_in - 1)
    y1 = np.minimum(y0 + 1, h_in - 1)
    
    wa = (x1 - x[inbounds]) * (y1 - y[inbounds])
    wb = (x[inbounds] - x0) * (y1 - y[inbounds])
    wc = (x1 - x[inbounds]) * (y[inbounds] - y0)
    wd = (x[inbounds] - x0) * (y[inbounds] - y0)

    imwarped = np.zeros((h_out, w_out) + ((im.shape[2],)), dtype=im.dtype)
    flat_out = imwarped.reshape(-1, im.shape[2])
    
    # add weighted average 
    flat_out[inbounds, :] = (wa[:, np.newaxis] * im[y0, x0, :] + 
                            wb[:, np.newaxis] * im[y0, x1, :] + 
                            wc[:, np.newaxis] * im[y1, x0, :] + 
                            wd[:, np.newaxis] * im[y1, x1, :])
    imwarped_nn = flat_out.reshape(h_out, w_out, im.shape[2])

    return imwarped_nn
 
def display_points_on_image(image, points, radius=30, output_path=None):
       
        img_with_points = image.copy()
        
        points_int = points.astype(int)
        
        for i, point in enumerate(points_int):
            x, y = point[0], point[1]
            cv2.circle(img_with_points, (x, y), radius, (255, 0, 0), -1)  
            cv2.putText(img_with_points, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        if output_path is not None:
            cv2.imwrite(output_path, cv2.cvtColor(img_with_points, cv2.COLOR_RGB2BGR))
            print(f"Image with points saved to: {output_path}")
        
        return img_with_points


def warp_points(points, H):
    points = np.asarray(points, dtype=float)
    if points.shape[1] != 2:
        raise ValueError("points must be (n,2).")
    n = points.shape[0]
    points = np.concatenate((points, np.ones((n, 1))), axis=1)

    points_transformed = H @ points.T
    points_transformed = points_transformed.T
    
    points_transformed = points_transformed[:, :2] / points_transformed[:, 2:3]
    
    return points_transformed

def blend_images(image, ref_image, image_points, ref_image_points, stiching_on_right):

    h1, w1 = image.shape[:2]
    h2, w2 = ref_image.shape[:2]
    
    # corners of the first image
    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])

    H = computeH(image_points, ref_image_points) 
    warped_image = warp_image_bilinear(image, H, stiching_on_right)

    corners1_transformed = warp_points(corners1, H)

    x_min = int(np.floor(np.min(corners1_transformed[:, 0])))
    y_min = int(np.floor(np.min(corners1_transformed[:, 1])))
    x_max = int(np.ceil(np.max(corners1_transformed[:, 0])))
    y_max = int(np.ceil(np.max(corners1_transformed[:, 1])))

    # required output size
    output_width = max(x_max, w2) - min(x_min, 0)
    output_height = max(y_max, h2) - min(y_min, 0)

    ref_image_canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    ref_image_canvas[-min(y_min, 0):-min(y_min, 0)+h2, -min(x_min, 0):-min(x_min, 0)+w2] = ref_image
    
    warped_image_canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # Calculate the offset to place the warped image on the canvas
    offset_x = -min(x_min, 0)
    offset_y = -min(y_min, 0)
    
    # dimensions of the warped image
    warped_h, warped_w = warped_image.shape[:2]    

    # use this for stiching on the right side
    if stiching_on_right:
        warped_image_canvas[offset_y:offset_y+warped_h, offset_x:offset_x+warped_w] = warped_image
    else:
        warped_image_canvas[:warped_h, :warped_w] = warped_image

    warped_image = warped_image_canvas

    warped_image_float = warped_image_canvas.astype(np.float32) / 255.0
    ref_image_float = ref_image_canvas.astype(np.float32) / 255.0

    # mask for overlapping regions
    mask = np.where((np.any(warped_image_canvas > 0, axis=2)) & (np.any(ref_image_canvas > 0, axis=2)), 0.5, 1.0)
    
    mask = np.stack([mask] * 3, axis=2)

    warped_masked = warped_image_float * mask
    ref_masked = ref_image_float * mask

    blended_image = warped_masked + ref_masked

    return blended_image



if __name__ == "__main__":
    # 1. load the image, last one is the reference image
    images = ['locker_right', 'locker_left_middle_blended'] # adjust filenames
    ims = []
    for img in images:
        i = Image.open(f'media/{img}.jpg')
        ims.append(np.asarray(i))

    im1_pts = np.loadtxt('im_pts_locker_right.txt')
    im2_pts = np.loadtxt('im_pts_locker_left_middle_blended.txt')

    blended_image = blend_images(ims[0], ims[1], im1_pts, im2_pts, stiching_on_right=True)
    plt.imshow(blended_image)
    plt.show()
    plt.close()

    plt.imsave('media/locker_left_middle_right_blended.jpg', blended_image)
