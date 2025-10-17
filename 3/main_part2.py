from PIL import Image
import numpy as np
from main import display_points_on_image, computeH, warp_points, blend_images
from harris import get_harris_corners
import matplotlib.pyplot as plt
import cv2
import os



def extract_axis_aligned_descriptors(img_gray, coords_yx, window_size=40, desc_size=8, gaussian_sigma=2.0, border=cv2.BORDER_REFLECT101, eps=1e-6):

    img = img_gray.astype(np.float32)
    if img.max() > 1.5:
        img /= 255.0

    M = coords_yx.shape[1]
    descs = []
    kept_idx = []

    for i in range(M):
        y = float(coords_yx[0, i])
        x = float(coords_yx[1, i])

        win = cv2.getRectSubPix(img, (window_size, window_size), (x, y))

        if gaussian_sigma is not None and gaussian_sigma > 0:
            k = int(gaussian_sigma * 6 + 1) | 1  # odd kernel
            win = cv2.GaussianBlur(win, (k, k), gaussian_sigma)

        patch8 = cv2.resize(win, (desc_size, desc_size), interpolation=cv2.INTER_AREA)

        vec = patch8.reshape(-1).astype(np.float32)
        m = vec.mean()
        s = vec.std()
        if s < eps:
            vec = np.zeros_like(vec, dtype=np.float32)
        else:
            vec = (vec - m) / (s + eps)

        descs.append(vec)
        kept_idx.append(i)

    if len(descs) == 0:
        return np.zeros((0, desc_size * desc_size), dtype=np.float32), np.array([], dtype=int)

    descs = np.vstack(descs).astype(np.float32)
    kept_idx = np.array(kept_idx, dtype=int)
    return descs, kept_idx


def anms_from_harris(harris_response, coords_yx, N=500, c_robust=0.9):

    ys = coords_yx[0].astype(int)
    xs = coords_yx[1].astype(int)
    strengths = harris_response[ys, xs].astype(np.float64)

    # sort descending for better lookup
    order = np.argsort(-strengths)
    pts = np.stack([ys[order], xs[order]], axis=1) 
    vals = strengths[order]
    M = len(vals)

    radii = np.full(M, np.inf, dtype=np.float64)

    for i in range(M):
        # indices of points that are sufficiently stronger
        stronger_idx = np.where(vals > vals[i] / c_robust)[0]
        if stronger_idx.size == 0:
            radii[i] = np.inf  # global maxima remain unsuppressed
            continue
        dy = pts[stronger_idx, 0] - pts[i, 0]
        dx = pts[stronger_idx, 1] - pts[i, 1]
        d2 = dx * dx + dy * dy
        radii[i] = np.sqrt(d2.min())

    # top-N by radius
    K = min(N, M)
    top = np.argsort(-radii)[:K]
    keep_idx_sorted_order = top

    # map back to original
    keep_idx_original = order[keep_idx_sorted_order]

    keep_coords_yx = coords_yx[:, keep_idx_original]
    keep_radii = radii[keep_idx_sorted_order]
    return keep_coords_yx, keep_radii, keep_idx_original




def match_descriptors_ratio(desc1, desc2, ratio_thresh=0.7, cross_check=False, return_scores=False):

    a2 = np.sum(desc1 * desc1, axis=1, keepdims=True)
    b2 = np.sum(desc2 * desc2, axis=1, keepdims=True).T
    d2 = np.maximum(a2 + b2 - 2.0 * (desc1 @ desc2.T), 0.0)

    # get index of the two closest descriptors
    nn1 = np.argpartition(d2, kth=1, axis=1)[:, :2]


    row_idx = np.arange(d2.shape[0])[:, None]
    best2_sorted = np.argsort(d2[row_idx, nn1], axis=1)
    nn1_sorted = nn1[row_idx, best2_sorted]

    j_best  = nn1_sorted[:, 0]
    j_second= nn1_sorted[:, 1]
    e1 = d2[row_idx[:,0], j_best]
    e2 = d2[row_idx[:,0], j_second]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = e1 / (e2 + 1e-12)

    keep = ratios < ratio_thresh
    i1_keep = np.nonzero(keep)[0]
    j2_keep = j_best[keep]

    if cross_check and i1_keep.size:
        j_best_rev = np.argmin(d2, axis=0)
        mutual = j_best_rev[j2_keep] == i1_keep
        i1_keep = i1_keep[mutual]
        j2_keep = j2_keep[mutual]
        ratios = ratios[keep][mutual]

    matches = np.stack([i1_keep, j2_keep], axis=1).astype(int)
    if return_scores:
        return matches, ratios[:matches.shape[0]]
    return matches


def ransac_homography(match_coords1, match_coords2, rounds=1000, threshold=5):
    #returns the inlier points and the homography matrix
    best_H = None
    best_inliers = 0
    best_inlier_mask = None

    for _ in range(rounds):
        sample = np.random.choice(match_coords1.shape[0], 4, replace=False)
        H = computeH(match_coords1[sample], match_coords2[sample])
        
        # calc distances for all points
        distances = np.abs(match_coords2 - warp_points(match_coords1, H))

        #use mask to check 
        inlier_mask = np.all(distances < threshold, axis=1)
        inlier_count = np.sum(inlier_mask)
        
        if inlier_count > best_inliers:
            best_inliers = inlier_count
            best_H = H
            best_inlier_mask = inlier_mask
    
    inlier_points1 = match_coords1[best_inlier_mask]
    inlier_points2 = match_coords2[best_inlier_mask]
    return (inlier_points1, inlier_points2), best_H


if __name__ == "__main__":
    scene = 'window'
    im1 = Image.open(f'media/{scene}_left_resized.jpg')
    im1_rgb = np.asarray(im1)
    im1 = np.asarray(im1.convert('L'))

    im2 = Image.open(f'media/{scene}_middle_resized.jpg')
    im2_rgb = np.asarray(im2)
    im2 = np.asarray(im2.convert('L'))

    im3 = Image.open(f'media/{scene}_right_resized.jpg')
    im3_rgb = np.asarray(im3)
    im3 = np.asarray(im3.convert('L'))

    h_im1, h_im1_coords = get_harris_corners(im1)
    h_im2, h_im2_coords = get_harris_corners(im2)
    h_im3, h_im3_coords = get_harris_corners(im3)

    print("Harris corners in image 1, 2, 3 done")

    h_im1_coords_display = h_im1_coords.T[:, [1, 0]]  
    h_im2_coords_display = h_im2_coords.T[:, [1, 0]]  
    h_im3_coords_display = h_im3_coords.T[:, [1, 0]]  

    im1_with_corners = display_points_on_image(im1_rgb, points=h_im1_coords_display, radius=3)
    im2_with_corners = display_points_on_image(im2_rgb, points=h_im2_coords_display, radius=3)

    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(im1_with_corners)
    plt.title(f'Kitchen Left - Harris Corners ({h_im1_coords.shape[1]} points)')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(im2_with_corners)
    plt.title(f'Kitchen Middle - Harris Corners ({h_im2_coords.shape[1]} points)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # cv2.imwrite('media/kitchen_left_harris_corners.jpg', cv2.cvtColor(im1_with_corners, cv2.COLOR_RGB2BGR))
    # cv2.imwrite('media/kitchen_middle_harris_corners.jpg', cv2.cvtColor(im2_with_corners, cv2.COLOR_RGB2BGR))
    
    # print("Images with Harris corners saved to media/ directory")


    # ANMS
    # paper suggests c_robust 0.9
    N_keep = 500
    c_robust = 0.9

    im1_anms_yx, im1_radii, _ = anms_from_harris(h_im1, h_im1_coords, N=N_keep, c_robust=c_robust)
    im2_anms_yx, im2_radii, _ = anms_from_harris(h_im2, h_im2_coords, N=N_keep, c_robust=c_robust)
    im3_anms_yx, im3_radii, _ = anms_from_harris(h_im3, h_im3_coords, N=N_keep, c_robust=c_robust)

    im1_anms_xy = im1_anms_yx.T[:, [1, 0]]
    im2_anms_xy = im2_anms_yx.T[:, [1, 0]]
    im3_anms_xy = im3_anms_yx.T[:, [1, 0]]

    im1_with_anms = display_points_on_image(im1_rgb.copy(), points=im1_anms_xy, radius=3)
    im2_with_anms = display_points_on_image(im2_rgb.copy(), points=im2_anms_xy, radius=3)

    desc1, idx1 = extract_axis_aligned_descriptors(im1, im1_anms_yx, window_size=40, desc_size=8)
    desc2, idx2 = extract_axis_aligned_descriptors(im2, im2_anms_yx, window_size=40, desc_size=8)
    desc3, idx3 = extract_axis_aligned_descriptors(im3, im3_anms_yx, window_size=40, desc_size=8)

    matches12, ratios = match_descriptors_ratio(desc1, desc2, ratio_thresh=0.7,
                                            cross_check=True, return_scores=True)

    # match_patch_output_dir = "media/match_patches"

    # num_matches_to_save = 5
    # for i in range(num_matches_to_save):
    #     best_match = matches12[i]
    #     idx1, idx2 = best_match[0], best_match[1]
    #     patch1 = desc1[idx1].reshape(8, 8)
    #     patch2 = desc2[idx2].reshape(8, 8)
    #     ratio = ratios[i] if len(ratios) > i else None

    #     plt.figure(figsize=(4, 2))
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(patch1, cmap='gray')
    #     plt.title('im1')
    #     plt.axis('off')
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(patch2, cmap='gray')
    #     plt.title('im2')
    #     plt.axis('off')
    #     plt.suptitle(f"Match {i+1}, Ratio: {ratio:.3f}" if ratio is not None else f"Match {i+1}")

    #     save_path = os.path.join(match_patch_output_dir, f"match_{i+1}_ratio_{ratio:.3f}.png" if ratio is not None else f"match_{i+1}.png")
    #     plt.tight_layout(pad=0.5)
    #     plt.savefig(save_path)
    #     plt.close()


    match_coords1 = im1_anms_xy[matches12[:, 0]]
    match_coords2 = im2_anms_xy[matches12[:, 1]]

    # plt.imshow(desc1[matches12[0, 0]].reshape(8, 8), cmap='gray')
    # plt.show()
    # win = cv2.getRectSubPix(im1, (40, 40), (int(match_coords1[0, 0]), int(match_coords1[0, 1])))
    # k = int(2.0 * 6 + 1) | 1  # odd kernel
    # win = cv2.GaussianBlur(win, (k, k), 2.0)
    # patch8 = cv2.resize(win, (8, 8), interpolation=cv2.INTER_AREA)
    # plt.imshow(patch8, cmap='gray')
    # plt.show()

    im1_with_matches = display_points_on_image(im1_rgb.copy(), points=match_coords1, radius=3)
    im2_with_matches = display_points_on_image(im2_rgb.copy(), points=match_coords2, radius=3)

    (im1_inliers, im2_inliers), H = ransac_homography(match_coords1, match_coords2)
    blended_image = blend_images(im1_rgb, im2_rgb, im1_inliers, im2_inliers, stiching_on_right=False)

    plt.imshow(blended_image)
    plt.show()
    plt.close()


    blended_image_rgb = (blended_image * 255.0).astype(np.uint8)
    blended_image = np.dot(blended_image[..., :3], [0.2989, 0.5870, 0.1140]).astype(blended_image.dtype)
    h_im_blended, h_im_blended_coords = get_harris_corners(blended_image, left_discard=2000)

    print("Harris corners in blended image done")


    im_blended_anms_yx, im_blended_anms_radii, _ = anms_from_harris(h_im_blended, h_im_blended_coords, N=N_keep, c_robust=c_robust)

    print("ANMS in blended image done")
    im_blended_anms_xy = im_blended_anms_yx.T[:, [1, 0]]

    im_blended_with_anms = display_points_on_image(blended_image_rgb, points=im_blended_anms_xy, radius=3)

    desc_blended, idx_blended = extract_axis_aligned_descriptors(blended_image, im_blended_anms_yx, window_size=40, desc_size=8)
    # match with warped image
    matches_blended, ratios_blended = match_descriptors_ratio(desc_blended, desc3, ratio_thresh=0.7,
                                            cross_check=True, return_scores=True)
    match_coords_blended = im_blended_anms_xy[matches_blended[:, 0]]
    match_coords3 = im3_anms_xy[matches_blended[:, 1]]

    (im3_inliers, im_warped_inliers), H = ransac_homography(match_coords3, match_coords_blended)
    blended_image123 = blend_images(im3_rgb, blended_image_rgb, im3_inliers, im_warped_inliers, True)
    
    plt.imshow(blended_image123)
    plt.show()
    plt.close()

    plt.imsave(f'media/{scene}_blended_image_auto_stiching.jpg', blended_image123)

    # plt.figure(figsize=(15, 6))
    # plt.subplot(1, 2, 1)
    # plt.imshow(im1_with_matches)
    # plt.title(f'Kitchen Left - Matches')
    # plt.axis('off')
    # plt.subplot(1, 2, 2)
    # plt.imshow(im2_with_matches)
    # plt.title(f'Kitchen Middle - Matches')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()
    # plt.imsave('media/kitchen_left_matches.jpg', im1_with_matches)
    # plt.imsave('media/kitchen_middle_matches.jpg', im2_with_matches)
 


    # plt.figure(figsize=(15, 6))
    # plt.subplot(1, 2, 1)
    # plt.imshow(im1_with_anms)
    # plt.title(f'Kitchen Left - ANMS Corners (N={im1_anms_xy.shape[0]})')
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.imshow(im2_with_anms)
    # plt.title(f'Kitchen Middle - ANMS Corners (N={im2_anms_xy.shape[0]})')
    # plt.axis('off')

    # plt.tight_layout()
    # plt.show()

    # cv2.imwrite('media/kitchen_left_anms_corners.jpg', cv2.cvtColor(im1_with_anms, cv2.COLOR_RGB2BGR))
    # cv2.imwrite('media/kitchen_middle_anms_corners.jpg', cv2.cvtColor(im2_with_anms, cv2.COLOR_RGB2BGR))
    # print("Images with ANMS corners saved to media/ directory")

        