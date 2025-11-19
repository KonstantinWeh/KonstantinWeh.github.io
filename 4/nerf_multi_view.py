import numpy as np


import matplotlib.pyplot as plt


def transform_c2w(c2w, x_c):
    c2w = np.asarray(c2w)
    x_c = np.asarray(x_c)

    ones = np.ones_like(x_c[..., :1])
    x_c_h = np.concatenate([x_c, ones], axis=-1)

    x_w_h = np.einsum("...ij,...j->...i", c2w, x_c_h)

    w = x_w_h[..., 3:4]
    return x_w_h[..., :3] / w


def pixel_to_camera(K, uv, s):
    K = np.asarray(K)
    uv = np.asarray(uv)
    s = np.asarray(s)

    K_inv = np.linalg.inv(K)

    ones = np.ones_like(uv[..., :1])
    uv_h = np.concatenate([uv, ones], axis=-1)

    scaled = uv_h * np.expand_dims(s, axis=-1)
    x_c = np.einsum("...ij,...j->...i", K_inv, scaled)
    return x_c


def pixel_to_ray(K, c2w, uv):
    K = np.asarray(K)
    c2w = np.asarray(c2w)
    uv = np.asarray(uv)

    depth = np.ones_like(uv[..., 0])
    x_c = pixel_to_camera(K, uv, depth)
    x_w = transform_c2w(c2w, x_c)

    ray_o = c2w[..., :3, 3]
    ray_d = x_w - ray_o

    norm = np.linalg.norm(ray_d, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    ray_d = ray_d / norm

    return ray_o, ray_d


def sample_random_rays(images, c2ws, focal_length, num_rays):
    images = np.asarray(images)
    c2ws = np.asarray(c2ws)

    n_images, height, width, _ = images.shape

    total_pixels = n_images * height * width
    indices = np.random.randint(0, total_pixels, size=num_rays)

    image_indices = indices // (height * width)
    pixel_indices = indices % (height * width)

    v = pixel_indices // width
    u = pixel_indices % width

    uv = np.stack([u + 0.5, v + 0.5], axis=-1)

    K = np.array([
        [focal_length, 0.0, width / 2.0],
        [0.0, focal_length, height / 2.0],
        [0.0, 0.0, 1.0],
    ])

    selected_c2ws = c2ws[image_indices]
    ray_o, ray_d = pixel_to_ray(K, selected_c2ws, uv)

    ray_colors = images[image_indices, v, u]

    return ray_o, ray_d, ray_colors, image_indices


def sample_points_along_rays(ray_o, ray_d, near=2.0, far=6.0, n_samples=64, perturb=False):
    ray_o = np.asarray(ray_o)
    ray_d = np.asarray(ray_d)

    prefix_shape = ray_o.shape[:-1]

    t_bins = np.linspace(near, far, n_samples + 1, dtype=ray_o.dtype)
    if prefix_shape:
        t_bins = np.reshape(t_bins, (1,) * len(prefix_shape) + (n_samples + 1,))
        t_bins = np.broadcast_to(t_bins, prefix_shape + (n_samples + 1,))

    lower = t_bins[..., :-1]
    upper = t_bins[..., 1:]

    if perturb:
        rand = np.random.rand(*prefix_shape, n_samples)
        t = lower + (upper - lower) * rand
    else:
        t = 0.5 * (lower + upper)

    points = ray_o[..., None, :] + ray_d[..., None, :] * t[..., :, None]

    return points, t





### provided 
data = np.load("media/lego/lego_200x200.npz")

# Training images: [100, 200, 200, 3]
images_train = data["images_train"] / 255.0

# Cameras for the training images
# (camera-to-world transformation matrix): [100, 4, 4]
c2ws_train = data["c2ws_train"]

# Validation images:
images_val = data["images_val"] / 255.0

# Cameras for the validation images: [10, 4, 4]
# (camera-to-world transformation matrix): [10, 200, 200, 3]
c2ws_val = data["c2ws_val"]

# Test cameras for novel-view video rendering:
# (camera-to-world transformation matrix): [60, 4, 4]
c2ws_test = data["c2ws_test"]

# Camera focal length
focal = data["focal"]  # float


class RaysData:
    def __init__(self, images, K, c2ws):
        self.images = np.asarray(images)
        self.K = np.asarray(K)
        self.c2ws = np.asarray(c2ws)
        self.N, self.H, self.W, _ = self.images.shape

        uu, vv = np.meshgrid(np.arange(self.W), np.arange(self.H), indexing="xy")
        per_image_uvs = np.stack([uu, vv], axis=-1).reshape(-1, 2)

        self.uvs = np.tile(per_image_uvs, (self.N, 1))
        self.pixels = self.images.reshape(-1, 3)
        self.image_indices = np.repeat(np.arange(self.N), self.H * self.W)

        uv_centers = self.uvs + 0.5
        sel_c2ws = self.c2ws[self.image_indices]
        ray_o, ray_d = pixel_to_ray(self.K, sel_c2ws, uv_centers)

        self.rays_o = ray_o.astype(np.float32)
        self.rays_d = ray_d.astype(np.float32)
        self.pixels = self.pixels.astype(np.float32)

    def sample_rays(self, num_rays):
        total = self.uvs.shape[0]
        idx = np.random.randint(0, total, size=num_rays)

        img_idx = self.image_indices[idx]
        uv_int = self.uvs[idx]
        uv = uv_int + 0.5  # shift to centers

        sel_c2w = self.c2ws[img_idx]

        ray_o, ray_d = pixel_to_ray(self.K, sel_c2w, uv)

        pixels = self.pixels[idx]

        return ray_o, ray_d, pixels




if __name__ == "__main__":
    import viser, time  # pip install viser
    import numpy as np

    if plt is None:
        raise ImportError("matplotlib is required for the visualization demo in nerf_multi_view.py")

    K = np.array([
        [focal, 0.0, images_train.shape[2] / 2.0],
        [0.0, focal, images_train.shape[1] / 2.0],
        [0.0, 0.0, 1.0],
    ])

    # --- You Need to Implement These ------
    dataset = RaysData(images_train, K, c2ws_train)
    rays_o, rays_d, pixels = dataset.sample_rays(100) # Should expect (B, 3)
    points, t = sample_points_along_rays(rays_o, rays_d, perturb=True)
    print("	points:", points.shape)
    print("t:", t.shape)
    print("colors:", np.zeros_like(points).reshape(-1, 3).shape)
    print("points:", points.reshape(-1, 3).shape)
    print("point_size:", 0.02)
    H, W = images_train.shape[1:3]



    # ## Visualization 1
    # server = viser.ViserServer(share=True)
    # for i, (image, c2w) in enumerate(zip(images_train, c2ws_train)):
    #     image_uint8 = (image * 255).astype(np.uint8)
    #     server.add_camera_frustum(
    #         f"/cameras/{i}",
    #         fov=2 * np.arctan2(H / 2, K[0, 0]),
    #         aspect=W / H,
    #         scale=0.15,
    #         wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
    #         position=c2w[:3, 3],
    #         image=image_uint8
    #     )
    # for i, (o, d) in enumerate(zip(rays_o, rays_d)):
    #     server.add_spline_catmull_rom(
    #         f"/rays/{i}", positions=np.stack((o, o + d * 6.0)),
    #     )
    # server.add_point_cloud(
    #     f"/samples",
    #     colors=np.zeros_like(points).reshape(-1, 3),
    #     points=points.reshape(-1, 3),
    #     point_size=0.02,
    # )

    # while True:
    #     time.sleep(0.1)  # Wait to allow visualization to run

    ## Visualization 2
    # This will check that your uvs aren't flipped
    uvs_start = 0
    uvs_end = 40000
    sample_uvs = dataset.uvs[uvs_start:uvs_end] # These are integer coordinates of widths / heights (xy not yx) of all the pixels in an image
    print("dataset.pixels:", dataset.pixels[uvs_start:uvs_end])
    plt.imshow(images_train[0, sample_uvs[:,1], sample_uvs[:,0]])
    plt.show()
    # uvs are array of xy coordinates, so we need to index into the 0th image tensor with [0, height, width], so we need to index with uv[:,1] and then uv[:,0]
    assert np.allclose(
        images_train[0, sample_uvs[:,1], sample_uvs[:,0]], 
        dataset.pixels[uvs_start:uvs_end], 
        rtol=0, atol=1e-2
    )

    # # Uncoment this to display random rays from the first image
    indices = np.random.randint(low=0, high=40000, size=100)

    # # Uncomment this to display random rays from the top left corner of the image
    # indices_x = np.random.randint(low=100, high=200, size=100)
    # indices_y = np.random.randint(low=0, high=100, size=100)
    # indices = indices_x + (indices_y * 200)

    data = {"rays_o": dataset.rays_o[indices], "rays_d": dataset.rays_d[indices]}
    points, t = sample_points_along_rays(data["rays_o"], data["rays_d"], perturb=True)
    # ---------------------------------------

    server = viser.ViserServer(share=False)
    for i, (image, c2w) in enumerate(zip(images_train, c2ws_train)):
        server.add_camera_frustum(
            f"/cameras/{i}",
            fov=2 * np.arctan2(H / 2, K[0, 0]),
            aspect=W / H,
            scale=0.15,
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
            image=image
        )
    for i, (o, d) in enumerate(zip(data["rays_o"], data["rays_d"])):
        positions = np.stack((o, o + d * 6.0))
        server.add_spline_catmull_rom(
            f"/rays/{i}", positions=positions,
        )
    server.add_point_cloud(
        f"/samples",
        colors=np.zeros_like(points).reshape(-1, 3),
        points=points.reshape(-1, 3),
        point_size=0.03,
    )

    while True:
        time.sleep(0.1)  # Wait to allow visualization to run




