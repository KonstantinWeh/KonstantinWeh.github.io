from PIL import Image
import numpy as np
from CS180_Project3_1 import warp_image_bilinear, warp_image_nearest_neighbor, computeH, get_points
import matplotlib.pyplot as plt


if __name__ == "__main__":
    image = Image.open('media/kitchen.jpg')
    image = np.asarray(image)

    # im1_pts, _ = get_points(image, image)
    # print("Saving correspondences to text files...")
    # np.savetxt('kitchen_im1_pts.txt', im1_pts)
     
    im1_pts = np.loadtxt('kitchen_im1_pts.txt')
    im2_pts = np.array([[200,1800], [1200,1800], [1200,200], [200,200]])
    H = computeH(im1_pts, im2_pts)
    
    im1_warped = warp_image_bilinear(image, H)
  
    plt.imshow(im1_warped)
    plt.imsave('media/kitchen_warped.jpg', im1_warped)
    plt.show()
    plt.close()