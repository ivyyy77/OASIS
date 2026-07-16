import os

import imageio.v3 as iio
import numpy as np
import torch


def images_to_video(images, output_path, fps, bitrate="10M"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames = []
    for image in images:
        if isinstance(images, torch.Tensor):
            image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        frames.append(image)
    iio.imwrite(
        output_path,
        np.stack(frames),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        bitrate=bitrate,
        macro_block_size=16,
    )
