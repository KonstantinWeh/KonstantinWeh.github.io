from CS180_Project3_1 import warp_image_nearest_neighbor, warp_image_bilinear
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from CS180_Project3_1 import display_points_on_image



 if __name__ == "__main__":

   images = ['left', 'middle_left', 'middle_right', 'right']
   ims = []
   for img in images:
      i = Image.open(f'media/{img}.jpg')
      i = i.transpose(Image.ROTATE_270)
      ims.append(np.asarray(i))

   im1_with_points = display_points_on_image(ims[0], im1_pts, 'media/im1_with_points.jpg')
   im2_with_points = display_points_on_image(ims[1], im2_pts, 'media/im2_with_points.jpg')


   import time
   print("\n=== Timing Analysis ===")
   
   start_time = time.time()
   im1_warped_nn = warp_image_nearest_neighbor(ims[0], H)
   nn_time = time.time() - start_time
   print(f"Nearest Neighbor Interpolation: {nn_time:.4f} seconds")
   
   start_time = time.time()
   im1_warped_bil = warp_image_bilinear(ims[0], H)
   bil_time = time.time() - start_time
   print(f"Bilinear Interpolation: {bil_time:.4f} seconds")
   
   print(f"Speed ratio (NN/Bilinear): {nn_time/bil_time:.2f}x")
   print(f"Bilinear is {bil_time/nn_time:.2f}x slower than Nearest Neighbor")

   # Display both warped images for comparison
   plt.figure(figsize=(15, 6))
   
   plt.subplot(1, 2, 1)
   plt.imshow(im1_warped_nn)
   plt.title(f'Nearest Neighbor ({nn_time:.4f}s)')
   plt.axis('off')
   
   plt.subplot(1, 2, 2)
   plt.imshow(im1_warped_bil)
   plt.title(f'Bilinear Interpolation ({bil_time:.4f}s)')
   plt.axis('off')
   
   plt.tight_layout()
   plt.savefig('media/interpolation_comparison.jpg', dpi=150, bbox_inches='tight')
   plt.show()
   plt.close()
   
   # Save individual warped images
   plt.imsave('media/im1_warped_nearest_neighbor.jpg', im1_warped_nn)
   plt.imsave('media/im1_warped_bilinear.jpg', im1_warped_bil)