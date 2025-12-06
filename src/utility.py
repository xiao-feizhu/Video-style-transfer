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

def compute_local_alpha(content_feat, eps=1e-5):
    # content_feat: (1, C, H, W)
    energy = torch.norm(content_feat, p=2, dim=1, keepdim=True)  # (1,1,H,W)
    e_min = energy.min()
    e_max = energy.max()
    alpha = (energy - e_min) / (e_max - e_min + eps)
    alpha = torch.softmax(alpha, dim=2)  # across H,W   
    return alpha  # 0~1

def combine_alpha(local_alpha, global_alpha, gamma=1.0):
    # alpha_global is scalar, local_alpha is (1,1,H,W)
    global_term = global_alpha ** gamma
    alpha_final = global_term * local_alpha
    return alpha_final.clamp(0,1)


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

def gaussian_blur(x, sigma=1.2):
    # x: (B,C,H,W)
    kernel_size = int(2 * round(3 * sigma) + 1)
    coords = torch.arange(kernel_size).float() - kernel_size//2
    g = torch.exp(-(coords**2)/(2*sigma*sigma))
    g = g / g.sum()
    kernel_1d = g.view(1,1,-1).to(x.device)
    kernel_2d = g.view(1,-1,1).to(x.device)

    # depthwise conv
    x = F.conv2d(x, kernel_1d.unsqueeze(0).repeat(x.size(1),1,1,1), padding=(kernel_size//2,0), groups=x.size(1))
    x = F.conv2d(x, kernel_2d.unsqueeze(0).repeat(x.size(1),1,1,1), padding=(0,kernel_size//2), groups=x.size(1))
    return x

def guided_filter(I, p, radius=6, eps=1e-3):
    """
    I: guidance image (B,C,H,W)
    p: filtering input (B,C,H,W)
    """
    B, C, H, W = I.shape
    # I.to('cuda')

    k = 2 * radius + 1
    # depthwise convolution kernel
    box = torch.ones((C, 1, k, k), device=I.device, dtype=I.dtype) / (k*k)

    # mean_I, mean_p
    mean_I = F.conv2d(I, box, padding=radius, groups=C)
    mean_p = F.conv2d(p, box, padding=radius, groups=C)

    # print(mean_I.device)
    # corr_I = E[I*I]
    corr_I = F.conv2d(I * I, box, padding=radius, groups=C)
    # corr_Ip = E[I*p]
    corr_Ip = F.conv2d(I * p, box, padding=radius, groups=C)

    var_I  = corr_I  - mean_I * mean_I
    cov_Ip = corr_Ip - mean_p * mean_I

    A = cov_Ip / (var_I + eps)
    b = mean_p - A * mean_I

    # mean_A, mean_b
    mean_A = F.conv2d(A, box, padding=radius, groups=C)
    mean_b = F.conv2d(b, box, padding=radius, groups=C)

    q = mean_A * I + mean_b
    return q


def unsharp_mask(x, amount=1.2, sigma=1.5):
    blur = gaussian_blur(x, sigma=sigma)
    return torch.clamp(x * (1 + amount) - blur * amount, 0, 1)

def tv_denoise(x, weight=1e-4):
    dh = x[:,:,1:,:] - x[:,:,:-1,:]
    dw = x[:,:,:,1:] - x[:,:,:,:-1]
    tv = torch.sum(torch.abs(dh)) + torch.sum(torch.abs(dw))
    return weight * tv

def postprocess_image(x):
    """
    x: Tensor (B,3,H,W)  in [0,1]
    return: processed x   in [0,1]
    """
    # 1. guided filter (去噪平滑 + 保边)
    x = guided_filter(x, x, radius=6, eps=1e-3)

    # 2. bilateral-like smoothing (强平滑)
    blur = gaussian_blur(x, sigma=1.0)
    x = 0.7 * x + 0.3 * blur

    # 3. sharpen (增强轮廓)
    # x = unsharp_mask(x, amount=0.8, sigma=1.0)

    return torch.clamp(x, 0, 1)

def normalize_frame(x):
    x = x.float() / 255.0
    # γ校正略微修复光线差异
    x = torch.clamp(x ** 0.95, 0, 1)
    return x

def match_histogram_torch(x, ref):
    x_flat = x.view(3, -1)
    ref_flat = ref.view(3, -1)
    # 简化版 normalize
    return torch.clamp((x - x.mean()) / (x.std()+1e-5) * ref.std() + ref.mean(), 0, 1)

def preprocess_for_flow(x):
    x = gaussian_blur(x, sigma=0.6)   # 小模糊
    return x

def preprocess_video_frame(frame, reference=None, prev_frame=None):
    """
    frame: (B,3,H,W), uint8
    reference: 第一帧的 normalized frame (可选)
    prev_frame: 上一帧 processed 后的结果 (可选)
    """
    frame = guided_filter(frame.float()/255.0, frame.float()/255.0, radius=5, eps=1e-3)

    frame = normalize_frame(frame)

    frame = preprocess_for_flow(frame)


    return frame

def occlusion_mask(flow_fw, flow_bw, threshold=1.0):
    """
    flow_fw: F_{t->t+1}
    flow_bw: F_{t+1->t}
    return mask in {0,1}
    """
    # Warp backward flow with forward flow
    flow_bw_warp = warp_with_flow(flow_bw, flow_fw)

    # Forward-backward consistency
    diff = flow_fw + flow_bw_warp
    mag = torch.sum(diff * diff, dim=1, keepdim=True)  # (B,1,H,W)

    # Thresholding — mask = 1 means reliable flow
    mask = (mag < threshold).float()
    return mask
