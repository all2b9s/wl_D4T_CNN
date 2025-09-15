from typing import Any
import numpy as np
import torch
from pixel_response import anacal_pix_r
from src.architecture.CNN_toolkit import shape_pixel_gradients, predictor, plot_shape_bidirectional, D4_eq_weight
from concurrent.futures import ProcessPoolExecutor

class calibrator():
    def __init__(self, model, device="cuda", workers=32, bs_para = [100,100]):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.workers = 32
        self.bs_size = bs_para[0]
        self.bs_times = bs_para[1]

    def __call__(self, q_imgs_m, 
                 q_imgs_1p, 
                 q_imgs_1n,
                 q_imgs_2p,
                 q_imgs_2n):
        """
        q_imgs_m,1p,1n,2p,2n: list of q_img dict for no-shear,1p,1n,2p,2n imgs each of shape (N, H, W)


        Returns:
        shapes_m: (N, 4) numpy array
        shapes_p: (N, 4) numpy array
        shapes_n: (N, 4) numpy array
        R_anacal: (N, 2, 2) numpy array
        R_delta: (N, 2, 2) numpy array
        """
        N = len(q_imgs_m)
        assert N == len(q_imgs_1p) and N == len(q_imgs_1n)

        # Get resmoothed shapes and gradients
        shapes_m, grads_e1_m, grads_e2_m = self.get_resmoothed_shape(q_imgs_m, with_grad=True)
        shapes_1p = self.get_resmoothed_shape(q_imgs_1p)
        shapes_1n = self.get_resmoothed_shape(q_imgs_1n)
        shapes_2p = self.get_resmoothed_shape(q_imgs_2p)
        shapes_2n = self.get_resmoothed_shape(q_imgs_2n)

        # Compute responses
        R_anacal_list = []
        R_delta_list = []
        for i in range(N):
            R_anacal, R_delta = _single_gal_response(q_imgs_m[i], grads_e1_m[i], grads_e2_m[i])
            R_anacal_list.append(R_anacal)
            R_delta_list.append(R_delta)

        R_anacal = np.array(R_anacal_list, dtype=np.float32)
        shear_1p = self.shear_measurement(shapes_1p)
        shear_1n = self.shear_measurement(shapes_1n)
        shear_2p = self.shear_measurement(shapes_2p)
        shear_2n = self.shear_measurement(shapes_2n)

        m1 = (shear_1p[0][0]-shear_1n[0][0])/R_anacal[:,0,0].mean()/2.0
        m2 = (shear_2p[0][1]-shear_2n[0][1])/R_anacal[:,1,1].mean()/2.0
        c1 = (shear_1p[0][0]+shear_1n[0][0])/2.0
        c2 = (shear_2p[0][1]+shear_2n[0][1])/2.0
        m1_err = (shear_1p[1][0]**2+shear_1n[1][0]**2)**0.5/R_anacal[:,0,0].mean()/2.0
        m2_err = (shear_2p[1][1]**2+shear_2n[1][1]**2)**0.5/R_anacal[:,1,1].mean()/2.0
        c1_err = (shear_1p[1][0]**2+shear_1n[1][0]**2)**0.5/2.0
        c2_err = (shear_2p[1][1]**2+shear_2n[1][1]**2)**0.5/2.0
        print(f"m1: {m1:.6f} +/- {m1_err:.6f}, m2: {m2:.6f} +/- {m2_err:.6f}, c1: {c1:.6f} +/- {c1_err:.6f}, c2: {c2:.6f} +/- {c2_err:.6f}")       

    def get_resmoothed_shape(self, q_imgs, batch_size=32, with_grad=False):
        """
        q_imgs: list of q_img dict, each of shape (5, H, W)
        batch_size: batch size for model inference
        with_grad: whether to compute pixel gradients

        Returns:
        if not with_grad:
            all_shape: (N, 2) numpy array
        else:
            all_shape: (N, 2) numpy array
            all_grad_e1: (N, H, W) numpy array
            all_grad_e2: (N, H, W) numpy array
        """

        def process_q_imgs(q_imgs_batch):
            resmoothed_batch = np.array([q_img[0].astype(np.float32)] for q_img in q_imgs_batch)  # shape (B, 1, H, W)
            if not with_grad:
                shapes = predictor(self.model, resmoothed_batch)
                grads_e1 = [None] * len(shapes)
                grads_e2 = [None] * len(shapes)
            else:
                shapes, grads_e1, grads_e2 = shape_pixel_gradients(self.model, resmoothed_batch)
            return shapes, grads_e1, grads_e2

        results = []
        for i in range(0, len(q_imgs), batch_size):
            q_imgs_batch = q_imgs[i:i + batch_size]
            results.append(process_q_imgs(q_imgs_batch))

        all_shape, all_grad_e1, all_grad_e2 = zip(*results)  # Unzip results
        all_shape = np.array([shape for batch in all_shape for shape in batch], dtype=np.float32)  # shape (N, 2)
        if not with_grad:
            return all_shape
        else:
            all_grad_e1 = np.array([grad for batch in all_grad_e1 for grad in batch], dtype=np.float32)
            all_grad_e2 = np.array([grad for batch in all_grad_e2 for grad in batch], dtype=np.float32)
            return all_shape, all_grad_e1, all_grad_e2  

    def prepare_q_images(self, images, psfs, noises, pixel_scale=0.2, sigma_arcsec=0.7/2.355, flim=10.0):
        """
        images: (N, H, W) numpy array
        psfs: (N, H, W) numpy array
        """
        N = images.shape[0]
        q_imgs = []

        def process_image(i):
            return anacal_pix_r(
                images[i,0],
                psfs[i,0],
                center=None,
                sigma_arcsec=sigma_arcsec,
                scale_arcsec_per_pix=pixel_scale,
                freq_lim=flim,
                noise_map=noises[i,0] if noises is not None else None
            )

        with ProcessPoolExecutor(self.workers) as executor:
            q_imgs = list(executor.map(process_image, range(N)))

        return q_imgs

    def shear_measurement(self, shapes):
        with ProcessPoolExecutor(self.workers) as executor:
            shape_futures = [executor.submit(_bootstrap, shapes, self.bs_size) for _ in range(self.bs_times)]
        shear_list = [future.result().mean(axis = 0) for future in shape_futures]
        shears = np.array(shear_list)
        return shears.mean(axis=0), shears.std(axis=0)
        
def _bootstrap(inputs, size):
    np.random.seed()
    rand_ints = np.random.choice(len(inputs), size=size, replace=True)
    return np.vstack([inputs[rand_index] for rand_index in rand_ints])

def _single_gal_response(q_img, grad_e1, grad_e2):
    """
    q_img: list of 5 q_img dict for 0,1p,1m,2p,2m each of shape (5, H, W)
    grad_e1, grad_e2: (H, W) numpy array
    """
    q_0, q_1p, q_1m, q_2p, q_2m = q_img # [v, g1, g2, j1, j2]
    delta_img1 = (q_1p[0].astype(np.float32) - q_1m[0].astype(np.float32)) / 2.0
    delta_img2 = (q_2p[0].astype(np.float32) - q_2m[0].astype(np.float32)) / 2.0

    anacal_R = np.zeros((2,2), dtype=np.float32)
    anacal_R[0,0] = np.sum(grad_e1 * q_0[1])
    anacal_R[0,1] = np.sum(grad_e1 * q_0[2])
    anacal_R[1,0] = np.sum(grad_e2 * q_0[1])
    anacal_R[1,1] = np.sum(grad_e2 * q_0[2])

    del_R = np.zeros((2,2), dtype=np.float32)
    del_R[0,0] = np.sum(grad_e1 * delta_img1)
    del_R[0,1] = np.sum(grad_e1 * delta_img2)
    del_R[1,0] = np.sum(grad_e2 * delta_img1)
    del_R[1,1] = np.sum(grad_e2 * delta_img2)

    return anacal_R, del_R



