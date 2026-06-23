import os
import argparse
import numpy as np
import cv2
from PIL import Image

import torch

try:
    from diffusers import AutoPipelineForInpainting
    _DIFFUSERS_AVAILABLE = True
    _DIFFUSERS_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001 - we want to catch any import/runtime error here
    AutoPipelineForInpainting = None
    _DIFFUSERS_AVAILABLE = False
    _DIFFUSERS_IMPORT_ERROR = e

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def read_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        # BGRA -> BGR, keep alpha separately when needed
        img = img[:, :, :3].copy()
    return img


def read_pikachu_bgr_and_alpha(path: str):
    """
    Returns:
      pik_bgr: uint8 BGR
      alpha: float32 [0,1], shape (H,W)
    If input has alpha channel, use it; otherwise, try GrabCut to extract.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")

    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alpha = alpha_from_grabcut(bgr)
        return bgr, alpha

    if img.shape[2] == 4:
        bgr = img[:, :, :3].copy()
        a_u8 = img[:, :, 3]
        alpha = a_u8.astype(np.float32) / 255.0
        return bgr, alpha

    # no alpha channel
    bgr = img
    alpha = alpha_from_grabcut(bgr)
    return bgr, alpha


def alpha_from_grabcut(bgr: np.ndarray, rect_scale=0.90, iter_count=5) -> np.ndarray:
    """Return alpha float32 in [0,1] assuming object roughly centered."""
    h, w = bgr.shape[:2]
    rw, rh = int(w * rect_scale), int(h * rect_scale)
    x = (w - rw) // 2
    y = (h - rh) // 2
    rect = (x, y, rw, rh)

    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(bgr, mask, rect, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)

    fg = cv2.GaussianBlur(fg, (0, 0), 1.5)
    fg = np.clip(fg, 0.0, 1.0)
    return fg


def resize_keep_aspect(img, target_w=None, target_h=None, scale=None, interp=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if scale is not None:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
    elif target_w is not None and target_h is None:
        s = target_w / float(w)
        new_w = target_w
        new_h = max(1, int(round(h * s)))
    elif target_h is not None and target_w is None:
        s = target_h / float(h)
        new_h = target_h
        new_w = max(1, int(round(w * s)))
    else:
        return img
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def paste_alpha_mask_on_canvas(canvas_hw, fg_alpha, x, y):
    """Place fg_alpha (H,W) onto a full canvas alpha (Hc,Wc), clipped."""
    Hc, Wc = canvas_hw
    h, w = fg_alpha.shape[:2]
    full = np.zeros((Hc, Wc), dtype=np.float32)

    x0 = max(0, x); y0 = max(0, y)
    x1 = min(Wc, x + w); y1 = min(Hc, y + h)
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Paste is out of bounds; adjust x/y or scale.")

    fx0 = x0 - x; fy0 = y0 - y
    fx1 = fx0 + (x1 - x0); fy1 = fy0 + (y1 - y0)

    full[y0:y1, x0:x1] = fg_alpha[fy0:fy1, fx0:fx1]
    return full


def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_bgr(img_pil: Image.Image) -> np.ndarray:
    rgb = np.array(img_pil)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def make_fg_mask_from_white_bg(img_path, out_mask_path="mask.png", tol=20, 
                               upscale = 10, scale=0.1):
    """
    从白色/近白背景图片生成前景二值掩码。
    tol: 背景“接近白色”的容差，越大越宽松（10~30 常用）
    输出: 256x256 的 mask.png (uint8, 0/255)，前景=255，背景=0
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    target_size = (256, 256)
    if img.shape[:2] != (target_size[1], target_size[0]):
        interp = cv2.INTER_AREA if img.shape[0] > 256 or img.shape[1] > 256 else cv2.INTER_CUBIC
        img = cv2.resize(img, target_size, interpolation=interp)

    h, w = img.shape[:2]
    # upscale = 4

    # 1) 先在高分辨率下分割，边界会比原图逐像素阈值更平滑、更贴边
    img_hr = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    lower = np.full(3, max(0, 255 - tol), dtype=np.uint8)
    upper = np.full(3, 255, dtype=np.uint8)
    bg = cv2.inRange(img_hr, lower, upper)   # 背景=255
    fg = cv2.bitwise_not(bg)                 # 前景=255

    # 2) 高分辨率下先做一次清理，去噪并填补边缘小裂缝
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * upscale + 1, 2 * upscale + 1)
    )
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, open_k, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, close_k, iterations=1)

    # 3) 只保留最大连通域，避免零散噪点影响最终轮廓
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (fg > 0).astype(np.uint8), connectivity=8
    )
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        fg = np.where(labels == largest, 255, 0).astype(np.uint8)
    else:
        fg = np.zeros_like(fg, dtype=np.uint8)

    # 4) 用有符号距离场做平滑，边界会更圆滑，同时比直接 blur 二值图更不容易偏移
    fg_bin = (fg > 0).astype(np.uint8)
    dist_in = cv2.distanceTransform(fg_bin, cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(1 - fg_bin, cv2.DIST_L2, 5)
    signed_dist = dist_in - dist_out
    signed_dist = cv2.GaussianBlur(
        signed_dist, (0, 0), sigmaX=scale * upscale, sigmaY=scale * upscale
    )
    fg = (signed_dist > 0).astype(np.uint8) * 255

    # 5) 回到 256x256，使用 area 下采样保持轮廓位置稳定，再阈值成二值掩码
    fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_AREA)
    fg = (fg >= 127).astype(np.uint8) * 255

    # 6) 最后做一次轻量 close，收一下边缘的小锯齿和微小缺口
    final_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, final_k, iterations=1)

    cv2.imwrite(out_mask_path, fg)
    return fg

if __name__ == "__main__":
    make_fg_mask_from_white_bg(
        img_path="./example_data/editing/masks/5-cat-new-only.png",        # 替换成你的图片路径
        out_mask_path="./example_data/editing/masks/5-cat-new_edit.png",
        # img_path="./example_data/editing/images/5-cat-new.png",        # 替换成你的图片路径
        # out_mask_path="./example_data/editing/images/5-cat_mask.png",
        # img_path="./example_data/text-to-avatar/masks/hellokitty-mask.png",        # 替换成你的图片路径
        # out_mask_path="./example_data/text-to-avatar/masks/hellokitty-mask-1.png",
        tol=30
    )


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--hand", required=True, help="hand image path")
#     ap.add_argument("--pikachu", required=True, help="pikachu image path (png/jpg)")
#     ap.add_argument("--out_dir", default="./out_sota", help="output directory")

#     # placement + size
#     ap.add_argument("--x", type=int, default=None, help="top-left x (default center)")
#     ap.add_argument("--y", type=int, default=None, help="top-left y (default center)")
#     ap.add_argument("--target_w", type=int, default=256, help="pikachu resized width (keep aspect)")
#     ap.add_argument("--scale", type=float, default=None, help="scale factor override")

#     # diffusion
#     ap.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0",
#                     help="SDXL base model id or local path")
#     ap.add_argument("--inpaint_model", default=None,
#                     help="optional SDXL inpaint model id/path; if None, uses AutoPipelineForInpainting on base")
#     ap.add_argument("--prompt", default="a cute pikachu sticker naturally placed on the hand, realistic lighting, high quality",
#                     help="positive prompt")
#     ap.add_argument("--negative", default="blurry, low quality, distorted, extra fingers, bad anatomy, watermark, text",
#                     help="negative prompt")
#     ap.add_argument("--steps", type=int, default=30)
#     ap.add_argument("--guidance", type=float, default=7.0)
#     ap.add_argument("--strength", type=float, default=0.85, help="inpaint strength (0-1)")
#     ap.add_argument("--seed", type=int, default=123)
#     ap.add_argument("--device", default="cuda", help="cuda or cpu")

#     args = ap.parse_args()
#     os.makedirs(args.out_dir, exist_ok=True)

#     # 1) load images
#     hand_bgr = read_bgr(args.hand)
#     pik_bgr, pik_alpha = read_pikachu_bgr_and_alpha(args.pikachu)

#     # 2) resize pikachu + alpha
#     if args.scale is not None:
#         pik_bgr = resize_keep_aspect(pik_bgr, scale=args.scale)
#         pik_alpha = resize_keep_aspect(pik_alpha, scale=args.scale, interp=cv2.INTER_LINEAR)
#     else:
#         pik_bgr = resize_keep_aspect(pik_bgr, target_w=args.target_w)
#         pik_alpha = resize_keep_aspect(pik_alpha, target_w=args.target_w, interp=cv2.INTER_LINEAR)

#     # 再保险一步：确保皮卡丘整体不会比手图更大，否则会被裁剪
#     H, W = hand_bgr.shape[:2]
#     h_pik, w_pik = pik_bgr.shape[:2]
#     if w_pik > W or h_pik > H:
#         fit_scale = min(W / float(w_pik), H / float(h_pik)) * 0.9
#         fit_scale = max(fit_scale, 1e-3)
#         pik_bgr = resize_keep_aspect(pik_bgr, scale=fit_scale)
#         pik_alpha = resize_keep_aspect(pik_alpha, scale=fit_scale, interp=cv2.INTER_LINEAR)

#     pik_alpha = np.clip(pik_alpha.astype(np.float32), 0.0, 1.0)

#     # 3) placement
#     h, w = pik_bgr.shape[:2]
#     x = (W - w) // 2 if args.x is None else args.x
#     y = (H - h) // 2 if args.y is None else args.y

#     # 4) build full-canvas mask in hand coords
#     alpha_full = paste_alpha_mask_on_canvas((H, W), pik_alpha, x, y)
#     # binary mask for inpaint (white=edit region)
#     mask_u8 = (alpha_full > 0.5).astype(np.uint8) * 255

#     # 5) build an initial "composite hint" (optional, helps inpaint keep the pikachu identity)
#     # We paste pikachu into hand as a starting image for inpainting.
#     hand_f = hand_bgr.astype(np.float32)
#     comp = hand_f.copy()
#     # paste region
#     x0 = max(0, x); y0 = max(0, y)
#     x1 = min(W, x + w); y1 = min(H, y + h)
#     fx0 = x0 - x; fy0 = y0 - y
#     fx1 = fx0 + (x1 - x0); fy1 = fy0 + (y1 - y0)

#     roi_a = pik_alpha[fy0:fy1, fx0:fx1][..., None]
#     roi_fg = pik_bgr[fy0:fy1, fx0:fx1].astype(np.float32)
#     roi_bg = comp[y0:y1, x0:x1]
#     comp[y0:y1, x0:x1] = roi_fg * roi_a + roi_bg * (1.0 - roi_a)
#     comp_u8 = np.clip(comp, 0, 255).astype(np.uint8)

#     # save mask + hint composite
#     cv2.imwrite(os.path.join(args.out_dir, "pikachu_mask.png"), mask_u8)
#     cv2.imwrite(os.path.join(args.out_dir, "composite_hint.png"), comp_u8)

#     # 6) optionally run SDXL inpainting; if diffusers/torchvision环境有问题，就直接用 composite 作为最终结果
#     if not _DIFFUSERS_AVAILABLE:
#         final_bgr = comp_u8
#         cv2.imwrite(os.path.join(args.out_dir, "final.png"), final_bgr)
#         print("[Warning] diffusers/torchvision not available, skipping inpainting.")
#         print("Reason:", repr(_DIFFUSERS_IMPORT_ERROR))
#     else:
#         try:
#             torch.manual_seed(args.seed)
#             generator = torch.Generator(device=args.device).manual_seed(args.seed)

#             model_id = args.inpaint_model if args.inpaint_model is not None else args.model
#             pipe = AutoPipelineForInpainting.from_pretrained(
#                 model_id,
#                 torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
#                 variant="fp16" if args.device.startswith("cuda") else None,
#             )
#             pipe = pipe.to(args.device)

#             image_pil = bgr_to_pil(comp_u8)       # start image with pikachu roughly placed
#             mask_pil = Image.fromarray(mask_u8)   # L mode 0/255

#             result = pipe(
#                 prompt=args.prompt,
#                 negative_prompt=args.negative,
#                 image=image_pil,
#                 mask_image=mask_pil,
#                 num_inference_steps=args.steps,
#                 guidance_scale=args.guidance,
#                 strength=args.strength,
#                 generator=generator
#             ).images[0]

#             final_bgr = pil_to_bgr(result)
#             cv2.imwrite(os.path.join(args.out_dir, "final.png"), final_bgr)
#         except Exception as e:  # noqa: BLE001
#             # 确保即使 inpainting 失败也能给出一个可用结果
#             final_bgr = comp_u8
#             cv2.imwrite(os.path.join(args.out_dir, "final.png"), final_bgr)
#             print("[Warning] Inpainting failed, using simple composite as final image.")
#             print("Reason:", repr(e))

#     print("Saved outputs to:", args.out_dir)
#     print("  final.png")
#     print("  pikachu_mask.png")
#     print("  composite_hint.png (debug)")


# if __name__ == "__main__":
#     main()
