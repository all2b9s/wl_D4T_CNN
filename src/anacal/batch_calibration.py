from typing import Any
import numpy as np
import torch
from src.anacal.pixel_response import anacal_pix_r
from src.architecture.CNN_toolkit import shape_pixel_gradients, predictor, plot_shape_bidirectional, D4_eq_weight
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import os

class Calibrator:
    def __init__(self, model, device="cuda", workers=32,
                 shear_value=0.02, dtype="float32"):
        """
        dtype: "float32" or "float64"
        """
        assert dtype in ("float32", "float64")
        self.dtype = dtype
        self.np_dtype = np.float64 if dtype == "float64" else np.float32
        self.torch_dtype = torch.float64 if dtype == "float64" else torch.float32

        self.model = model.to(device=device, dtype=self.torch_dtype)
        self.model.eval()

        self.device = device
        self.workers = workers
        self.shear_value = shear_value


    # ---------------------------- shape_measure ----------------------------
    def shape_measure(self, q_imgs, batch_size=32):
        N = len(q_imgs)
        assert hasattr(q_imgs, "shape") and q_imgs.shape[1] == 5, \
            "q_imgs must be shape (N, 5, H, W)"

        shapes, grad_e1, grad_e2 = \
            self.get_resmoothed_shape(q_imgs, batch_size=batch_size, with_grad=True)

        # compute response per galaxy
        res = [
            _single_gal_response(q_imgs[i], grad_e1[i], grad_e2[i])
            for i in range(N)
        ]

        R_anacal_list = np.stack(res, axis=0).astype(self.np_dtype)

        return shapes, R_anacal_list


    # ---------------------------- get_resmoothed_shape ----------------------------
    def get_resmoothed_shape(self, q_imgs, batch_size=32, with_grad=False):

        def process_q_imgs(q_imgs_batch):
            # Extract the "m" image (index 0) and cast to target dtype
            resmoothed_batch = np.array(
                [q_img[0].astype(self.np_dtype) for q_img in q_imgs_batch]
            )  # shape (B, H, W)

            # ensure shape is (B, 1, H, W)
            if resmoothed_batch.ndim == 3:
                resmoothed_batch = resmoothed_batch[:, None, :, :]

            # Move to torch
            inp = torch.from_numpy(resmoothed_batch).to(
                device=self.device, dtype=self.torch_dtype
            )

            if not with_grad:
                out = predictor(self.model, inp)[:, :2]  # (B, 2)
                shapes = out.detach().cpu().numpy().astype(self.np_dtype)
                return shapes, [None]*len(shapes), [None]*len(shapes)

            else:
                shapes_t, g1_t, g2_t = shape_pixel_gradients(self.model, inp)

                shapes = shapes_t.astype(self.np_dtype)
                g1 = g1_t.astype(self.np_dtype)
                g2 = g2_t.astype(self.np_dtype)
                return shapes, g1, g2


        # batching
        results = []
        for i in range(0, len(q_imgs), batch_size):
            batch = q_imgs[i:i+batch_size]
            results.append(process_q_imgs(batch))

        # unpack
        all_shape, all_grad_e1, all_grad_e2 = zip(*results)

        all_shape = np.concatenate(all_shape, axis=0).astype(self.np_dtype)

        if not with_grad:
            return all_shape

        all_grad_e1 = np.concatenate(all_grad_e1, axis=0).astype(self.np_dtype)
        all_grad_e2 = np.concatenate(all_grad_e2, axis=0).astype(self.np_dtype)

        return all_shape, all_grad_e1, all_grad_e2

def _process_image(
    i: int,
    images: np.ndarray,
    psfs: np.ndarray,
    centers: np.ndarray,
    noises: np.ndarray,
    pixel_scale: float,
    sigma_arcsec: float,
    flim: float,
):
    # your existing per-index logic:
    return anacal_pix_r(
        images[i],
        psfs[i],
        center=centers[i],
        scale_arcsec_per_pix=pixel_scale,
        sigma_arcsec=sigma_arcsec,
        freq_lim=flim,
        noise_map=(noises[i] if noises is not None else None),
    )


# ---- Bootstraping ----
_BS_GLOBALS = {"shapes": None, "Rs": None, "shear": None}

def _bs_init_pool(shapes, Rs, shear_value):
    # Set read-only globals inside each worker process
    _BS_GLOBALS["shapes"] = shapes
    _BS_GLOBALS["Rs"]     = Rs
    _BS_GLOBALS["shear"]  = shear_value

def _bs_worker(bs_ind):
    # bs_ind: 1D numpy index array
    shapes = _BS_GLOBALS["shapes"][bs_ind]
    Rs     = _BS_GLOBALS["Rs"][bs_ind]
    m, c   = _calibration(shapes, Rs)   # expects to return (m, c) shaped (2,), (2,)
    shear  = _BS_GLOBALS["shear"]
    return (m / shear - 1.0, c)         # normalize m here to reduce post work

def get_biases(shapes, Rs,
                shear_value=0.02,
               bs_size=100, 
               bs_times=100,
               n_factor=1.0,
               is_twin = False,
               n_jobs=4):
    """
    shapes: (N, 4, 2) numpy array 
    Rs:     (N, 4, 2, 2) numpy array   # combined R inputs you use in _calibration
    Returns:
        m_mean (2,), c_mean (2,), m_std (2,), c_std (2,)
        where m_mean is already (m/shear - 1)
    """
    N = shapes.shape[0]
    # Pre-generate bootstrap indices
    if is_twin:
        N = N // 2
    bs_inds = [_bootstrap(N, bs_size) for _ in range(bs_times)]
    if is_twin:
        bs_inds = [np.concatenate([inds, inds + N]) for inds in bs_inds]

    # Parallel map over bootstrap resamples
    if n_jobs is None:
        n_jobs = os.cpu_count() or 1

    # Initialize worker pool with shared read-only arrays and shear value
    with ProcessPoolExecutor(max_workers=n_jobs,
                             initializer=_bs_init_pool,
                             initargs=(shapes, Rs, shear_value)) as ex:
        results = list(ex.map(_bs_worker, bs_inds))

    # Unpack results
    m_list = np.array([r[0] for r in results], dtype=np.float64)  # (bs_times, 2)
    c_list = np.array([r[1] for r in results], dtype=np.float64)  # (bs_times, 2)

    # Full-sample estimate (single call, no bootstrap)
    m_mean, c_mean = _calibration(shapes, Rs)
    m_mean = m_mean / shear_value - 1.0

    # Bootstrap standard deviations
    m_std = m_list.std(axis=0, ddof=1) * n_factor
    c_std = c_list.std(axis=0, ddof=1) * n_factor

    return m_mean, c_mean, m_std, c_std


# --- globals inside workers ---
_G = {}
def _init_worker(images, psfs, centers, noises, pixel_scale, sigma_arcsec, flim):
    # mark read-only to avoid copy-on-write if using fork
    for arr in (images, psfs, centers, noises):
        try:
            arr.setflags(write=False)
        except Exception:
            pass
    _G['images'] = images
    _G['psfs'] = psfs
    _G['centers'] = centers
    _G['noises'] = noises
    _G['pixel_scale'] = pixel_scale
    _G['sigma_arcsec'] = sigma_arcsec
    _G['flim'] = flim

def _process_idx(i):
    # call your existing function with globals
    return _process_image(
        i,
        images=_G['images'],
        psfs=_G['psfs'],
        centers=_G['centers'],
        noises=_G['noises'],
        pixel_scale=_G['pixel_scale'],
        sigma_arcsec=_G['sigma_arcsec'],
        flim=_G['flim'],
    )

def prepare_q_images(images, psfs, noises,
                     cat=None, pixel_scale=0.2,
                     sigma_arcsec=0.85/2.355, flim=10.0, workers=32):

    N = images.shape[0]
    centers = np.ones((N, 2), dtype=np.float64) * (images.shape[1] // 2)
    if cat is not None:
        if len(cat) > N:
            cat = cat[:N]  # ensure cat length matches N
            print(f"Warning: cat length greater than images. Truncating cat to length {N}.")
        centers[:, 0] += (cat['image_x']-(cat['image_x']+0.5)//1).to_numpy(np.float64)
        centers[:, 1] += (cat['image_y']-(cat['image_y']+0.5)//1).to_numpy(np.float64)

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(images, psfs, centers, noises, pixel_scale, sigma_arcsec, flim),
    ) as ex:
        # key changes: use ex.map on indices ONLY + chunksize + tqdm
        q_imgs = list(
            tqdm(
                ex.map(_process_idx, range(N), chunksize=16),
                total=N,
                desc="prepare_q_images",
            )
        )
    return np.array(q_imgs)


def _calibration(shapes, Rs):
    """
    shapes: (N, 4, 2) numpy array 
    R: (N, 4, 2, 2) numpy array

    all in sequence of 1p,1n,2p,2n

    Returns:
    cal_shapes: (N, 4) numpy array
    """
    shears = shapes.mean(axis=0) # (4,2)
    responses = Rs.mean(axis=0) # (4, 2, 2)

    m1 = (shears[0,0]-shears[1,0])/(responses[0,0,0]+responses[1,0,0])
    m2 = (shears[2,1]-shears[3,1])/(responses[2,1,1]+responses[3,1,1])
    c1 = (shears[0,0]+shears[1,0])/(responses[0,0,0]+responses[1,0,0])
    c2 = (shears[2,1]+shears[3,1])/(responses[2,1,1]+responses[3,1,1])

    return np.array([m1, m2]), np.array([c1, c2])
        
def _bootstrap(lense, size, seed = None):
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()
    rand_ints = np.random.choice(lense, size=size, replace=True)
    return rand_ints

def _single_gal_response(q_img, grad_e1, grad_e2):
    """
    q_img: list of 5 q_img dict for 0,1p,1m,2p,2m each of shape (5, H, W)
    grad_e1, grad_e2: (H, W) numpy array
    """

    anacal_R = np.zeros((2,2), dtype=q_img.dtype)
    anacal_R[0,0] = np.sum(grad_e1 * q_img[1])
    anacal_R[0,1] = np.sum(grad_e1 * q_img[2])
    anacal_R[1,0] = np.sum(grad_e2 * q_img[1])
    anacal_R[1,1] = np.sum(grad_e2 * q_img[2])


    return np.array([anacal_R])



