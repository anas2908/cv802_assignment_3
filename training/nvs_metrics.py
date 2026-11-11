"""Image-quality metrics used for every method (same code for all, so the numbers are comparable).

PSNR  = 10 log10(1 / MSE)                                     (higher is better)
SSIM  = structural similarity, 11x11 Gaussian window, sigma 1.5 (Wang et al. 2004; higher is better)
LPIPS = learned perceptual distance with VGG features (Zhang et al. 2018; lower is better)
Images are float tensors (3, H, W) in [0, 1].
"""
import math

import torch
import torch.nn.functional as F

_lpips = None


def psnr(img, gt):
    mse = torch.mean((img - gt) ** 2).item()
    return 10 * math.log10(1.0 / max(mse, 1e-12))


def ssim(img, gt, window=11, sigma=1.5):
    g = torch.exp(-(torch.arange(window, dtype=torch.float32, device=img.device) - window // 2) ** 2 / (2 * sigma ** 2))
    g = (g / g.sum())
    w = (g[:, None] @ g[None, :]).expand(3, 1, window, window).contiguous()
    x, y = img[None], gt[None]
    mu_x, mu_y = F.conv2d(x, w, padding=window // 2, groups=3), F.conv2d(y, w, padding=window // 2, groups=3)
    sxx = F.conv2d(x * x, w, padding=window // 2, groups=3) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=window // 2, groups=3) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=window // 2, groups=3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return m.mean().item()


def lpips(img, gt):
    global _lpips
    if _lpips is None:
        import lpips as lp
        _lpips = lp.LPIPS(net="vgg", verbose=False).to(img.device).eval()
    with torch.no_grad():
        return _lpips(img[None] * 2 - 1, gt[None] * 2 - 1).item()


def all_metrics(img, gt):
    return {"psnr": psnr(img, gt), "ssim": ssim(img, gt), "lpips": lpips(img, gt)}
