from CS180_Project3_1 import get_points
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    images = ['locker_left_middle_blended','locker_right'] # adjust filenames

    ims = []
    for img in images:
        i = Image.open(f'media/{img}.jpg')
        # i = i.transpose(Image.ROTATE_90)
        # h, w = np.asarray(i).shape[:2]
        # i = i.resize((w//3, h//3))
        ims.append(np.asarray(i))
        
    
    im1_pts, im2_pts = get_points(ims[0], ims[1], 8)

    
    # print("Saving correspondences to text files...")
    np.savetxt('im_pts_locker_left_middle_blended.txt', im1_pts)
    np.savetxt('im_pts_locker_right.txt', im2_pts)
    # print("Arrays exported successfully!")
    # print("Files created:")
    # print("- im1_pts_couch_left.txt, im2_pts_couch_middle.txt (text format)")