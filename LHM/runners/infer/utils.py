






import glob
import json
import math
import os
from collections import defaultdict

import pickle
import cv2
try:
    import decord
except ImportError:
    decord = None
import numpy as np
import torch
from PIL import Image
from pytorch3d.io import save_ply
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
import trimesh

def generate_rotation_matrix_y(degrees):
    theta = math.radians(degrees)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    R = [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]

    return np.asarray(R, dtype=np.float32)


def scale_intrs(intrs, ratio_x, ratio_y):
    if len(intrs.shape) >= 3:
        intrs[:, 0] = intrs[:, 0] * ratio_x
        intrs[:, 1] = intrs[:, 1] * ratio_y
    else:
        intrs[0] = intrs[0] * ratio_x
        intrs[1] = intrs[1] * ratio_y
    return intrs


def calc_new_tgt_size(cur_hw, tgt_size, multiply):
    ratio = tgt_size / min(cur_hw)
    tgt_size = int(ratio * cur_hw[0]), int(ratio * cur_hw[1])
    tgt_size = (
        int(tgt_size[0] / multiply) * multiply,
        int(tgt_size[1] / multiply) * multiply,
    )
    ratio_y, ratio_x = tgt_size[0] / cur_hw[0], tgt_size[1] / cur_hw[1]
    return tgt_size, ratio_y, ratio_x


def calc_new_tgt_size_by_aspect(cur_hw, aspect_standard, tgt_size, multiply):
    assert abs(cur_hw[0] / cur_hw[1] - aspect_standard) < 0.03
    tgt_size = tgt_size * aspect_standard, tgt_size
    tgt_size = (
        int(tgt_size[0] / multiply) * multiply,
        int(tgt_size[1] / multiply) * multiply,
    )
    ratio_y, ratio_x = tgt_size[0] / cur_hw[0], tgt_size[1] / cur_hw[1]
    return tgt_size, ratio_y, ratio_x


def _load_pose(pose):
    intrinsic = torch.eye(4)
    intrinsic[0, 0] = pose["focal"][0]
    intrinsic[1, 1] = pose["focal"][1]
    intrinsic[0, 2] = pose["princpt"][0]
    intrinsic[1, 2] = pose["princpt"][1]
    intrinsic = intrinsic.float()

    c2w = torch.eye(4)


    c2w = c2w.float()

    return c2w, intrinsic


def _load_pose_sign(pose, cam_trans=None):

    batch_size = pose["focal"].shape[0]


    intrinsic = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)


    intrinsic[:, 0, 0] = pose["focal"][:, 0]
    intrinsic[:, 1, 1] = pose["focal"][:, 1]
    intrinsic[:, 0, 2] = pose["princpt"][:, 0]
    intrinsic[:, 1, 2] = pose["princpt"][:, 1]


    c2w = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)

    if cam_trans is not None:
        c2w[:, :3, 3] = cam_trans






    return c2w, intrinsic


def img_center_padding(img_np, pad_ratio):

    ori_w, ori_h = img_np.shape[:2]

    w = round((1 + pad_ratio) * ori_w)
    h = round((1 + pad_ratio) * ori_h)

    if len(img_np.shape) > 2:
        img_pad_np = np.zeros((w, h, img_np.shape[2]), dtype=np.uint8)
    else:
        img_pad_np = np.zeros((w, h), dtype=np.uint8)
    offset_h, offset_w = (w - img_np.shape[0]) // 2, (h - img_np.shape[1]) // 2
    img_pad_np[
        offset_h : offset_h + img_np.shape[0] :, offset_w : offset_w + img_np.shape[1]
    ] = img_np

    return img_pad_np


def resize_image_keepaspect_np(img, max_tgt_size):
    """
    similar to ImageOps.contain(img_pil, (img_size, img_size)) # keep the same aspect ratio
    """
    h, w = img.shape[:2]
    ratio = max_tgt_size / max(h, w)
    new_h, new_w = round(h * ratio), round(w * ratio)
    return cv2.resize(img, dsize=(new_w, new_h), interpolation=cv2.INTER_AREA)


def center_crop_according_to_mask(img, mask, aspect_standard, enlarge_ratio):
    """
    img: [H, W, 3]
    mask: [H, W]
    """
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        raise Exception("empty mask")

    x_min = np.min(xs)
    x_max = np.max(xs)
    y_min = np.min(ys)
    y_max = np.max(ys)

    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2

    half_w = max(abs(center_x - x_min), abs(center_x - x_max))
    half_h = max(abs(center_y - y_min), abs(center_y - y_max))
    half_w_raw = half_w
    half_h_raw = half_h
    aspect = half_h / half_w

    if aspect >= aspect_standard:
        half_w = round(half_h / aspect_standard)
    else:
        half_h = round(half_w * aspect_standard)


    if half_h > center_y:
        half_w = round(half_h_raw / aspect_standard)
        half_h = half_h_raw
    if half_w > center_x:
        half_h = round(half_w_raw * aspect_standard)
        half_w = half_w_raw

    if abs(enlarge_ratio[0] - 1) > 0.01 or abs(enlarge_ratio[1] - 1) > 0.01:
        enlarge_ratio_min, enlarge_ratio_max = enlarge_ratio
        enlarge_ratio_max_real = min(center_y / half_h, center_x / half_w)
        enlarge_ratio_max = min(enlarge_ratio_max_real, enlarge_ratio_max)
        enlarge_ratio_min = min(enlarge_ratio_max_real, enlarge_ratio_min)
        enlarge_ratio_cur = (
            np.random.rand() * (enlarge_ratio_max - enlarge_ratio_min)
            + enlarge_ratio_min
        )
        half_h, half_w = round(enlarge_ratio_cur * half_h), round(
            enlarge_ratio_cur * half_w
        )

    assert half_h <= center_y
    assert half_w <= center_x
    assert abs(half_h / half_w - aspect_standard) < 0.03

    offset_x = center_x - half_w
    offset_y = center_y - half_h

    new_img = img[offset_y : offset_y + 2 * half_h, offset_x : offset_x + 2 * half_w]
    new_mask = mask[offset_y : offset_y + 2 * half_h, offset_x : offset_x + 2 * half_w]

    return new_img, new_mask, offset_x, offset_y


def preprocess_image(
    rgb_path,
    mask_path,
    intr,
    pad_ratio,
    bg_color,
    max_tgt_size,
    aspect_standard,
    enlarge_ratio,
    render_tgt_size,
    multiply,
    need_mask=True,
):
    """inferece
    image, _, _ = preprocess_image(image_path, mask_path=None, intr=None, pad_ratio=0, bg_color=1.0,
                                        max_tgt_size=896, aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1.0],
                                        render_tgt_size=source_size, multiply=14, need_mask=True)

    """

    rgb = np.array(Image.open(rgb_path))
    rgb_raw = rgb.copy()
    if pad_ratio > 0:
        rgb = img_center_padding(rgb, pad_ratio)

    rgb = rgb / 255.0
    if need_mask:
        if rgb.shape[2] < 4:
            if mask_path is not None:
                mask = np.array(Image.open(mask_path))
            else:
                from rembg import remove


                mask = remove(rgb_raw[:, :, (2, 1, 0)])[:, :, -1]
                print("rmbg mask: ", mask.min(), mask.max(), mask.shape)
            if pad_ratio > 0:
                mask = img_center_padding(mask, pad_ratio)
            mask = mask / 255.0
        else:

            assert rgb.shape[2] == 4
            mask = rgb[:, :, 3]
    else:

        mask = np.ones_like(rgb[:, :, 0])

    mask = (mask > 0.5).astype(np.float32)
    rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])


    rgb = resize_image_keepaspect_np(rgb, max_tgt_size)
    mask = resize_image_keepaspect_np(mask, max_tgt_size)


    rgb, mask, offset_x, offset_y = center_crop_according_to_mask(
        rgb, mask, aspect_standard, enlarge_ratio
    )
    if intr is not None:
        intr[0, 2] -= offset_x
        intr[1, 2] -= offset_y



    tgt_hw_size, ratio_y, ratio_x = calc_new_tgt_size_by_aspect(
        cur_hw=rgb.shape[:2],
        aspect_standard=aspect_standard,
        tgt_size=render_tgt_size,
        multiply=multiply,
    )

    rgb = cv2.resize(
        rgb, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
    )
    mask = cv2.resize(
        mask, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
    )

    if intr is not None:


        intr = scale_intrs(intr, ratio_x=ratio_x, ratio_y=ratio_y)
        assert (
            abs(intr[0, 2] * 2 - rgb.shape[1]) < 2.5
        ), f"{intr[0, 2] * 2}, {rgb.shape[1]}"
        assert (
            abs(intr[1, 2] * 2 - rgb.shape[0]) < 2.5
        ), f"{intr[1, 2] * 2}, {rgb.shape[0]}"


        intr[0, 2] = rgb.shape[1] // 2
        intr[1, 2] = rgb.shape[0] // 2

    rgb = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
    mask = (
        torch.from_numpy(mask[:, :, None]).float().permute(2, 0, 1).unsqueeze(0)
    )
    return rgb, mask, intr


def extract_imgs_from_video(video_file, save_root, fps):
    if decord is None:
        raise ImportError("decord is required to extract frames from video inputs.")
    print(f"extract_imgs_from_video:{video_file}")
    vr = decord.VideoReader(video_file)
    for i in range(0, len(vr), fps):
        frame = vr[i].asnumpy()
        save_path = os.path.join(save_root, f"{i:05d}.jpg")
        cv2.imwrite(save_path, frame[:, :, (2, 1, 0)])


def predict_motion_seqs_from_images(image_folder: str, save_root, fps=6):
    id_name = os.path.splitext(os.path.basename(image_folder))[0]
    if os.path.isfile(image_folder) and (
        image_folder.endswith("mp4") or image_folder.endswith("move")
    ):
        save_frame_root = os.path.join(save_root, "extracted_frames", id_name)
        if not os.path.exists(save_frame_root):
            os.makedirs(save_frame_root, exist_ok=True)
            extract_imgs_from_video(
                video_file=image_folder, save_root=save_frame_root, fps=fps
            )
        else:
            print("skip extract_imgs_from_video......")
        image_folder = save_frame_root

    image_folder_abspath = os.path.abspath(image_folder)
    print(f"predict motion seq:{image_folder_abspath}")
    save_smplx_root = image_folder + "_smplx_params_mhmr"
    if not os.path.exists(save_smplx_root):
        cmd = f"cd thirdparty/multi-hmr &&  python infer_batch.py  --data_root {image_folder_abspath}  --out_folder {image_folder_abspath} --crop_head   --crop_hand   --pad_ratio 0.2 --smplify"
        os.system(cmd)
    else:
        print("skip predict smplx.........")
    return save_smplx_root, image_folder


def prepare_motion_seqs(
    motion_seqs_dir,
    image_folder,
    save_root,
    fps,
    bg_color,
    aspect_standard,
    enlarge_ratio,
    render_image_res,
    need_mask,
    multiply=16,
    motion_size=3000,
):
    """
    Prepare motion sequences for rendering.

    Args:
        motion_seqs_dir (str): Directory path of motion sequences.
        image_folder (str): Directory path of source images.
        save_root (str): Directory path to save the motion sequences.
        fps (int): Frames per second for the motion sequences.
        bg_color (tuple): Background color in RGB format.
        aspect_standard (float): Standard human aspect ratio (height/width).
        enlarge_ratio (float): Ratio to enlarge the source images.
        render_image_res (int): Resolution of the rendered images.
        need_mask (bool): Flag indicating whether masks are needed.
        multiply (int, optional): Multiply factor for image size. Defaults to 16.

    Returns:
        dict: Dictionary containing the prepared motion sequences.
            'render_c2ws': camera to world matrix  [B F 4 4]
            'render_intrs': intrins matrix [B F 4 4]
            'render_bg_colors' bg_colors [B F 3]
            'smplx_params': smplx_params -> ['betas',
                        'root_pose', 'body_pose',
                        'jaw_pose', 'leye_pose',
                        'reye_pose', 'lhand_pose',
                        'rhand_pose',
                        'trans', 'expr', 'focal',
                        'princpt',
                        'img_size_wh'
                ]
            'rgbs': imgs w.r.t motions

    Raises:
        AssertionError: If motion_seqs_dir is None and image_folder is None.

    """

    if motion_seqs_dir is None:
        assert image_folder is not None
        motion_seqs_dir, image_folder = predict_motion_seqs_from_images(
            image_folder, save_root, fps
        )

    motion_seqs = sorted(glob.glob(os.path.join(motion_seqs_dir, "*.json")))
    motion_seqs = motion_seqs[:motion_size]


    c2ws, intrs, rgbs, bg_colors, masks = [], [], [], [], []
    smplx_params = []
    shape_param = None

    for idx, smplx_path in enumerate(motion_seqs):

        if image_folder is not None:
            file_name = os.path.splitext(os.path.basename(smplx_path))[0]
            frame_path = os.path.join(image_folder, file_name + ".png")
            if not os.path.exists(frame_path):
                frame_path = os.path.join(image_folder, file_name + ".jpg")
        with open(smplx_path) as f:
            smplx_raw_data = json.load(f)
            smplx_param = {
                k: torch.FloatTensor(v)
                for k, v in smplx_raw_data.items()
                if "pad_ratio" not in k
            }

        if idx == 0:
            shape_param = smplx_param["betas"]

        c2w, intrinsic = _load_pose(smplx_param)
        intrinsic_raw = intrinsic.clone()
        if "expr" not in smplx_raw_data:

            max_tgt_size = int(max(smplx_param["img_size_wh"]))
        else:
            max_tgt_size = int(smplx_param["img_size_wh"][0])
            smplx_param.pop("expr")

        flame_path = smplx_path.replace("smplx_params", "flame_params")
        smplx_param["expr"] = torch.FloatTensor([0.0] * 100)

        smplx_param["expr"] = torch.FloatTensor([0.0] * 100)

        c2ws.append(c2w)
        bg_colors.append(bg_color)
        intrs.append(intrinsic)

        smplx_params.append(smplx_param)

    c2ws = torch.stack(c2ws, dim=0)
    intrs = torch.stack(intrs, dim=0)
    bg_colors = (
        torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)
    )

    if len(rgbs) > 0:
        rgbs = torch.cat(rgbs, dim=0)


    smplx_params_tmp = defaultdict(list)
    for smplx in smplx_params:
        for k, v in smplx.items():
            smplx_params_tmp[k].append(v)
    for k, v in smplx_params_tmp.items():
        smplx_params_tmp[k] = torch.stack(v)
    smplx_params = smplx_params_tmp

    smplx_params["betas"] = shape_param


    for k, v in smplx_params.items():
        smplx_params[k] = v.unsqueeze(0)

    c2ws = c2ws.unsqueeze(0)
    intrs = intrs.unsqueeze(0)
    bg_colors = bg_colors.unsqueeze(0)
    if len(rgbs) > 0:
        rgbs = rgbs.unsqueeze(0)


    motion_seqs_ret = {}
    motion_seqs_ret["render_c2ws"] = c2ws
    motion_seqs_ret["render_intrs"] = intrs
    motion_seqs_ret["render_bg_colors"] = bg_colors
    motion_seqs_ret["smplx_params"] = smplx_params
    motion_seqs_ret["rgbs"] = rgbs
    motion_seqs_ret["motion_seqs"] = motion_seqs

    return motion_seqs_ret


def prepare_sign_motion_seqs(
    motion_seqs,
    image_folder,
    save_root,
    fps,
    bg_color,
    aspect_standard,
    enlarge_ratio,
    render_image_res,
    need_mask,
    multiply=16,
    motion_size=3000,
):
    """
    Prepare motion sequences for rendering.

    Args:
        motion_seqs_dir (str): Directory path of motion sequences.
        image_folder (str): Directory path of source images.
        save_root (str): Directory path to save the motion sequences.
        fps (int): Frames per second for the motion sequences.
        bg_color (tuple): Background color in RGB format.
        aspect_standard (float): Standard human aspect ratio (height/width).
        enlarge_ratio (float): Ratio to enlarge the source images.
        render_image_res (int): Resolution of the rendered images.
        need_mask (bool): Flag indicating whether masks are needed.
        multiply (int, optional): Multiply factor for image size. Defaults to 16.

    Returns:
        dict: Dictionary containing the prepared motion sequences.
            'render_c2ws': camera to world matrix  [B F 4 4]
            'render_intrs': intrins matrix [B F 4 4]
            'render_bg_colors' bg_colors [B F 3]
            'smplx_params': smplx_params -> ['betas',
                        'root_pose', 'body_pose',
                        'jaw_pose', 'leye_pose',
                        'reye_pose', 'lhand_pose',
                        'rhand_pose',
                        'trans', 'expr', 'focal',
                        'princpt',
                        'img_size_wh'
                ]
            'rgbs': imgs w.r.t motions

    Raises:
        AssertionError: If motion_seqs_dir is None and image_folder is None.

    """






























    results_dict = motion_seqs
    smplx_params = {}

    focals = results_dict['focal']
    princpts = results_dict['princpt']

    all_pose = results_dict['smplx']
    if all_pose.shape[0] != 289:
        print('1')
    img_size = torch.tensor([720, 1280])
    pred_focals = torch.tensor([1330.2149658203125, 1330.2149658203125])
    pred_princpts = torch.tensor([768, 432])



    all_pose = torch.tensor(all_pose)
    bg_colors = torch.tensor(bg_color, dtype=torch.float32).unsqueeze(-1).repeat(all_pose.size(0),3)



    focals = torch.tensor(focals)
    princpts = torch.tensor(princpts)
    focals = focals[0, :].repeat(all_pose.size(0), 1)
    princpts = princpts[0, :].repeat(all_pose.size(0), 1)




    smplx_params['root_pose'] = all_pose[:, :3]
    smplx_params['body_pose'] = all_pose[:, 3:66].reshape(-1, 21, 3)
    smplx_params['lhand_pose'] = all_pose[:, 66:111].reshape(-1, 15, 3)
    smplx_params['rhand_pose'] = all_pose[:, 111:156].reshape(-1, 15, 3)
    smplx_params['jaw_pose'] = all_pose[:, 156:159]
    smplx_params['betas'] = all_pose[0, 159:169]











    smplx_params['expr'] = torch.cat([all_pose[:, 169:179], torch.zeros(all_pose.size(0), 90)], dim=1)


    smplx_params['trans'] = all_pose[0, 179:182].repeat(all_pose.size(0), 1)

    smplx_params['leye_pose'] = torch.zeros_like(smplx_params['jaw_pose'])
    smplx_params['reye_pose'] = torch.zeros_like(smplx_params['jaw_pose'])
    smplx_params['focal'] = focals
    smplx_params['princpt'] = princpts

    smplx_params['img_size_wh'] = img_size.repeat(all_pose.size(0), 1)


















    cam_trans = None
    c2ws, intrs = _load_pose_sign(smplx_params, cam_trans)






    for k, v in smplx_params.items():
        smplx_params[k] = v.unsqueeze(0)
    c2ws = c2ws.unsqueeze(0)
    intrs = intrs.unsqueeze(0)

    rgbs = []
    motion_seqs_ret = {}
    motion_seqs_ret["render_c2ws"] = c2ws
    motion_seqs_ret["render_intrs"] = intrs

    motion_seqs_ret["render_bg_colors"] = bg_colors.unsqueeze(0)
    motion_seqs_ret["smplx_params"] = smplx_params
    motion_seqs_ret["rgbs"] = rgbs


    return motion_seqs_ret



def prepare_motion_single(
    motion_seqs_dir,
    image_name,
    save_root,
    fps,
    bg_color,
    aspect_standard,
    enlarge_ratio,
    render_image_res,
    need_mask,
    multiply=16,
):
    """
    Prepare motion sequences for rendering.

    Args:
        motion_seqs_dir (str): Directory path of motion sequences.
        image_folder (str): Directory path of source images.
        save_root (str): Directory path to save the motion sequences.
        fps (int): Frames per second for the motion sequences.
        bg_color (tuple): Background color in RGB format.
        aspect_standard (float): Standard human aspect ratio (height/width).
        enlarge_ratio (float): Ratio to enlarge the source images.
        render_image_res (int): Resolution of the rendered images.
        need_mask (bool): Flag indicating whether masks are needed.
        multiply (int, optional): Multiply factor for image size. Defaults to 16.

    Returns:
        dict: Dictionary containing the prepared motion sequences.
            'render_c2ws': camera to world matrix  [B F 4 4]
            'render_intrs': intrins matrix [B F 4 4]
            'render_bg_colors' bg_colors [B F 3]
            'smplx_params': smplx_params -> ['betas',
                        'root_pose', 'body_pose',
                        'jaw_pose', 'leye_pose',
                        'reye_pose', 'lhand_pose',
                        'rhand_pose',
                        'trans', 'expr', 'focal',
                        'princpt',
                        'img_size_wh'
                ]
            'rgbs': imgs w.r.t motions

    Raises:
        AssertionError: If motion_seqs_dir is None and image_folder is None.

    """

    motion_seqs = [
        os.path.join(
            motion_seqs_dir,
            image_name.replace(".jpg", ".png").replace(".png", ".json"),
        )
        for _ in range(4)
    ]
    axis_list = [0, 60, 180, 240]


    c2ws, intrs, rgbs, bg_colors, masks = [], [], [], [], []
    smplx_params = []
    shape_param = None

    for idx, smplx_path in enumerate(motion_seqs):

        with open(smplx_path) as f:
            smplx_raw_data = json.load(f)
            smplx_param = {
                k: torch.FloatTensor(v)
                for k, v in smplx_raw_data.items()
                if "pad_ratio" not in k
            }

        if idx == 0:
            shape_param = smplx_param["betas"]

        c2w, intrinsic = _load_pose(smplx_param)
        intrinsic_raw = intrinsic.clone()
        if "expr" not in smplx_raw_data:

            max_tgt_size = int(max(smplx_param["img_size_wh"]))
        else:
            max_tgt_size = int(smplx_param["img_size_wh"][0])
            smplx_param.pop("expr")

        flame_path = smplx_path.replace("smplx_params", "flame_params")
        if os.path.exists(flame_path):
            with open(flame_path) as f:
                flame_param = json.load(f)
                smplx_param["expr"] = torch.FloatTensor(flame_param["expcode"])


                smplx_param["jaw_pose"] = torch.FloatTensor(flame_param["posecode"][3:])
                smplx_param["leye_pose"] = torch.FloatTensor(flame_param["eyecode"][:3])
                smplx_param["reye_pose"] = torch.FloatTensor(flame_param["eyecode"][3:])

        else:
            smplx_param["expr"] = torch.FloatTensor([0.0] * 100)

        root_rotate_matrix = axis_angle_to_matrix(smplx_param["root_pose"])
        rotate = generate_rotation_matrix_y(axis_list[idx])
        rotate = torch.from_numpy(rotate).float()
        rotate = rotate @ root_rotate_matrix
        new_rotate_axis = matrix_to_axis_angle(rotate)

        smplx_param["root_pose"] = new_rotate_axis

        c2ws.append(c2w)
        bg_colors.append(bg_color)
        intrs.append(intrinsic)

        smplx_params.append(smplx_param)

    c2ws = torch.stack(c2ws, dim=0)
    intrs = torch.stack(intrs, dim=0)
    bg_colors = (
        torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)
    )

    if len(rgbs) > 0:
        rgbs = torch.cat(rgbs, dim=0)


    smplx_params_tmp = defaultdict(list)
    for smplx in smplx_params:
        for k, v in smplx.items():
            smplx_params_tmp[k].append(v)
    for k, v in smplx_params_tmp.items():
        smplx_params_tmp[k] = torch.stack(v)
    smplx_params = smplx_params_tmp

    smplx_params["betas"] = shape_param


    for k, v in smplx_params.items():
        smplx_params[k] = v.unsqueeze(0)

    c2ws = c2ws.unsqueeze(0)
    intrs = intrs.unsqueeze(0)
    bg_colors = bg_colors.unsqueeze(0)
    if len(rgbs) > 0:
        rgbs = rgbs.unsqueeze(0)


    motion_seqs = {}
    motion_seqs["render_c2ws"] = c2ws
    motion_seqs["render_intrs"] = intrs
    motion_seqs["render_bg_colors"] = bg_colors
    motion_seqs["smplx_params"] = smplx_params
    motion_seqs["rgbs"] = rgbs

    return motion_seqs


def prepare_motion_lrmbench(
    motion_seqs_dir,
    image_name,
    save_root,
    fps,
    bg_color,
    aspect_standard,
    enlarge_ratio,
    render_image_res,
    need_mask,
    multiply=16,
):
    """
    Prepare motion sequences for rendering.

    Args:
        motion_seqs_dir (str): Directory path of motion sequences.
        image_folder (str): Directory path of source images.
        save_root (str): Directory path to save the motion sequences.
        fps (int): Frames per second for the motion sequences.
        bg_color (tuple): Background color in RGB format.
        aspect_standard (float): Standard human aspect ratio (height/width).
        enlarge_ratio (float): Ratio to enlarge the source images.
        render_image_res (int): Resolution of the rendered images.
        need_mask (bool): Flag indicating whether masks are needed.
        multiply (int, optional): Multiply factor for image size. Defaults to 16.

    Returns:
        dict: Dictionary containing the prepared motion sequences.
            'render_c2ws': camera to world matrix  [B F 4 4]
            'render_intrs': intrins matrix [B F 4 4]
            'render_bg_colors' bg_colors [B F 3]
            'smplx_params': smplx_params -> ['betas',
                        'root_pose', 'body_pose',
                        'jaw_pose', 'leye_pose',
                        'reye_pose', 'lhand_pose',
                        'rhand_pose',
                        'trans', 'expr', 'focal',
                        'princpt',
                        'img_size_wh'
                ]
            'rgbs': imgs w.r.t motions

    Raises:
        AssertionError: If motion_seqs_dir is None and image_folder is None.

    """
    import json

    axis_list = list(range(0, 360, 30))

    motion_seqs = [
        os.path.join(
            motion_seqs_dir,
            image_name.replace(".jpg", ".png")
            .replace(".jpeg", ".png")
            .replace(".PNG", ".png")
            .replace(".png", ".json"),
        )
        for _ in range(len(axis_list))
    ]


    c2ws, intrs, rgbs, bg_colors, masks = [], [], [], [], []
    smplx_params = []
    shape_param = None

    for idx, smplx_path in enumerate(motion_seqs):

        with open(smplx_path) as f:
            smplx_raw_data = json.load(f)
            smplx_param = {
                k: torch.FloatTensor(v)
                for k, v in smplx_raw_data.items()
                if "pad_ratio" not in k
            }

        if idx == 0:
            shape_param = smplx_param["betas"]

        c2w, intrinsic = _load_pose(smplx_param)
        intrinsic_raw = intrinsic.clone()
        if "expr" not in smplx_raw_data:

            max_tgt_size = int(max(smplx_param["img_size_wh"]))
        else:
            max_tgt_size = int(smplx_param["img_size_wh"][0])
            smplx_param.pop("expr")

        flame_path = smplx_path.replace("smplx_params", "flame_params")
        if os.path.exists(flame_path):
            with open(flame_path) as f:
                flame_param = json.load(f)
                smplx_param["expr"] = torch.FloatTensor(flame_param["expcode"])


                smplx_param["jaw_pose"] = torch.FloatTensor(flame_param["posecode"][3:])
                smplx_param["leye_pose"] = torch.FloatTensor(flame_param["eyecode"][:3])
                smplx_param["reye_pose"] = torch.FloatTensor(flame_param["eyecode"][3:])

        else:
            smplx_param["expr"] = torch.FloatTensor([0.0] * 100)

        smplx_param["expr"] = torch.FloatTensor([0.0] * 100)

        root_rotate_matrix = axis_angle_to_matrix(smplx_param["root_pose"])
        rotate = generate_rotation_matrix_y(axis_list[idx])
        rotate = torch.from_numpy(rotate).float()
        rotate = rotate @ root_rotate_matrix
        new_rotate_axis = matrix_to_axis_angle(rotate)

        smplx_param["root_pose"] = new_rotate_axis

        c2ws.append(c2w)
        bg_colors.append(bg_color)
        intrs.append(intrinsic)

        smplx_params.append(smplx_param)

    c2ws = torch.stack(c2ws, dim=0)
    intrs = torch.stack(intrs, dim=0)
    bg_colors = (
        torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)
    )

    if len(rgbs) > 0:
        rgbs = torch.cat(rgbs, dim=0)


    smplx_params_tmp = defaultdict(list)
    for smplx in smplx_params:
        for k, v in smplx.items():
            smplx_params_tmp[k].append(v)
    for k, v in smplx_params_tmp.items():
        smplx_params_tmp[k] = torch.stack(v)
    smplx_params = smplx_params_tmp

    smplx_params["betas"] = shape_param


    for k, v in smplx_params.items():
        smplx_params[k] = v.unsqueeze(0)

    c2ws = c2ws.unsqueeze(0)
    intrs = intrs.unsqueeze(0)
    bg_colors = bg_colors.unsqueeze(0)
    if len(rgbs) > 0:
        rgbs = rgbs.unsqueeze(0)

    motion_seqs = {}
    motion_seqs["render_c2ws"] = c2ws
    motion_seqs["render_intrs"] = intrs
    motion_seqs["render_bg_colors"] = bg_colors
    motion_seqs["smplx_params"] = smplx_params
    motion_seqs["rgbs"] = rgbs

    return motion_seqs
