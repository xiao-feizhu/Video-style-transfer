import math
import torch
import torch.nn.functional as F
from model import calc_mean_std, adain

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_for_vgg(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize input [0,1] to ImageNet stats for VGG.
    """
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)

def denorm_from_vgg(x: torch.Tensor) -> torch.Tensor:
    x = x * IMAGENET_STD.to(x.device) + IMAGENET_MEAN.to(x.device)
    return torch.clamp(x, 0.0, 1.0)

# def style_interpolation(encoder, content_feat, style_imgs, weights):
#     """
#     style_imgs: list of style image tensors
#     weights: list of floats that sum to 1
#     """
#     assert math.isclose(sum(weights), 1.0, rel_tol=1e-5)

#     style_feats = []
#     for img in style_imgs:
#         sf = encoder(normalize_for_vgg(img))
#         style_feats.append(sf[-1])

#     # compute weighted mean/std
#     means = []
#     stds = []
#     for sf, w in zip(style_feats, weights):
#         m, s = calc_mean_std(sf)
#         means.append(m * w)
#         stds.append(s * w)

#     interp_mean = sum(means)
#     interp_std = sum(stds)

#     c_mean, c_std = calc_mean_std(content_feat[-1])
#     normalized = (content_feat[-1] - c_mean) / c_std
#     return normalized * interp_std + interp_mean

def compute_local_alpha(content_feat, eps=1e-5):
    # content_feat: (1, C, H, W)
    energy = torch.norm(content_feat, p=2, dim=1, keepdim=True)  # (1,1,H,W)
    e_min = energy.min()
    e_max = energy.max()
    alpha = (energy - e_min) / (e_max - e_min + eps)
    return alpha  # 0~1

def combine_alpha(local_alpha, global_alpha, gamma=1.0):
    # alpha_global is scalar, local_alpha is (1,1,H,W)
    global_term = global_alpha ** gamma
    alpha_final = global_term * local_alpha
    return alpha_final.clamp(0,1)

def style_interpolation(c4, style_feats, weights):
    """
    style_feats: list of s4 tensors [4 tensors]
    weights: list or tensor of shape (4,)
    """
    assert math.isclose(sum(weights), 1.0, rel_tol=1e-5)
    t = 0
    for w, s4 in zip(weights, style_feats):
        t = t + w * adain(c4, s4)
    return t

def interpolate_weights(n=5, device="cuda"):
    """
    4-style 2D interpolation → n×n 坐标。
    输出 shape = (n*n, 4)
    """
    coords = torch.linspace(0, 1, n, device=device)
    weights = []

    for i in range(n):
        for j in range(n):
            w1 = (1 - coords[i]) * (1 - coords[j])
            w2 = (1 - coords[i]) * coords[j]
            w3 = coords[i] * (1 - coords[j])
            w4 = coords[i] * coords[j]
            w = torch.tensor([w1, w2, w3, w4], device=device)
            w = w / w.sum()
            weights.append(w)

    return torch.stack(weights)   # (n*n, 4)


def match_color(content_img, stylized_img):
    """
    Replace the LAB color of stylized image with content image.
    This preserves color while keeping style textures.
    """
    import cv2
    c = content_img.squeeze().permute(1,2,0).cpu().numpy()
    s = stylized_img.squeeze().permute(1,2,0).cpu().numpy()

    c = (c*255).astype("uint8")
    s = (s*255).astype("uint8")

    c_lab = cv2.cvtColor(c, cv2.COLOR_RGB2LAB)
    s_lab = cv2.cvtColor(s, cv2.COLOR_RGB2LAB)

    # keep L channel from stylized, ab from content
    out_lab = s_lab.copy()
    out_lab[:,:,1:] = c_lab[:,:,1:]

    out_rgb = cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)
    out_rgb = torch.tensor(out_rgb/255.).float().permute(2,0,1).unsqueeze(0)
    return out_rgb


def warp_with_flow(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warp image img according to optical flow.
    img:  [B, C, H, W]
    flow: [B, 2, H, W], flow in pixels (dx, dy) from prev -> current
    """
    B, C, H, W = img.shape

    # create base grid
    ys, xs = torch.meshgrid(
        torch.arange(0, H, device=img.device),
        torch.arange(0, W, device=img.device),
        indexing="ij",
    )
    xs = xs.float()
    ys = ys.float()

    # add flow
    flow_x = flow[:, 0, :, :]
    flow_y = flow[:, 1, :, :]

    x = xs[None, :, :] + flow_x
    y = ys[None, :, :] + flow_y

    # normalize to [-1, 1] for grid_sample
    x = 2.0 * x / max(W - 1, 1) - 1.0
    y = 2.0 * y / max(H - 1, 1) - 1.0

    grid = torch.stack((x, y), dim=-1)  # [B, H, W, 2]
    warped = F.grid_sample(
        img, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return warped
