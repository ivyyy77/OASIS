"""
Generate per-frame InterHand annotations (anno/img/mask) from combined anno_cam.pkl.

Usage:
  # process all frames in framelist
  python scripts/generate_interhand_anno.py \
      --dataset-root /path/to/InterHand \
      --subject test/Capture0/ROM03_RT_No_Occlusion \
      --phase test

  # process a single frame
  python scripts/generate_interhand_anno.py \
      --dataset-root /path/to/InterHand \
      --subject test/Capture0/ROM03_RT_No_Occlusion \
      --phase test \
      --frame test/Capture0/ROM03_RT_No_Occlusion/cam400272/image15012.jpg
"""

import os
import argparse
import pickle
import re

import cv2
import numpy as np

from tools_utils.model.ohta.configs import cfg
from data.utils.augm_util import augmentation, trans_point2d
from data.utils.camera_util import apply_global_tfm_to_camera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-root',
        default=os.environ.get('INTERHAND_ROOT'),
        help='Root path of InterHand dataset'
    )
    parser.add_argument(
        '--output-root',
        default='./example_data/interhand2.6m',
        help='Output directory for generated annotations, images, and masks'
    )
    parser.add_argument(
        '--subject',
        default='test/Capture0/ROM03_RT_No_Occlusion',
        help='Subject folder'
    )
    parser.add_argument(
        '--phase',
        default='test',
        help='train | test | infer'
    )
    parser.add_argument(
        '--frame',
        default=None,
        help='Optional single frame (relative path). If not set, process all frames.'
    )

    args = parser.parse_args()
    if not args.dataset_root:
        parser.error('--dataset-root is required when INTERHAND_ROOT is not set')

    dataset_root = args.dataset_root
    subject = args.subject

    image_dir = os.path.join(
        dataset_root,
        f'InterHand2.6M_{cfg.interhand.fps}fps_batch1/images'
    )

    if args.phase == 'train' or subject.startswith('train/'):
        anno_name = os.path.join(
            image_dir.replace('images', 'preprocess_ohta_our_full'),
            'train/prior_learning_data',
            'anno_cam.pkl'
        )
    else:
        anno_name = os.path.join(
            image_dir.replace('images', 'preprocess'),
            subject,
            'anno_cam.pkl'
        )

    if not os.path.exists(anno_name):
        print('[ERROR] Combined anno not found:', anno_name)
        return

    print('[INFO] Loading combined annotation:', anno_name)
    with open(anno_name, 'rb') as f:
        cameras, mesh_infos, bbox, framelists = pickle.load(f)

    framelist = framelists[::200]
    print(len(framelist))

    if len(framelist) == 0:
        print('[ERROR] Empty framelist')
        return

    if args.frame is not None:
        rel = args.frame
        if rel not in framelist:
            jpg_idx = rel.find('.jpg')
            rel_trunc = rel[:jpg_idx + 4] if jpg_idx != -1 else rel
            if rel_trunc in framelist:
                rel = rel_trunc
            else:
                name = os.path.splitext(os.path.basename(rel))[0]
                cands = [f for f in framelist if name in f]
                if not cands:
                    print('[ERROR] Frame not found in framelist:', args.frame)
                    return
                rel = cands[0]
        frames_to_process = [rel]
    else:
        frames_to_process = list(framelist)

    print(f'[INFO] Will process {len(frames_to_process)} frames')

    base_out = args.output_root
    out_anno = os.path.join(base_out, 'anno')
    out_img = os.path.join(base_out, 'images')
    out_mask = os.path.join(base_out, 'masks')

    os.makedirs(out_anno, exist_ok=True)
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_mask, exist_ok=True)

    written = 0

    for frame in frames_to_process:
        safe = frame.replace('/', '_').replace('.jpg', '')
        out_pkl = os.path.join(out_anno, f'{safe}.pkl')

        cam_orig = cameras.get(frame)
        mesh_orig = mesh_infos.get(frame)
        bbox_orig = bbox.get(frame)

        if bbox_orig is None:
            continue

        img_path = os.path.join(image_dir, frame)
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        mask_path = img_path.replace('/images/', '/masks_removeblack/').replace('.jpg', '.png')
        alpha = None
        if os.path.exists(mask_path):
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                alpha = m

        bgcolor = np.zeros(3, dtype=np.float32)

        try:
            img_aug, img2bb_trans, _, aug_param, _, _, alpha_aug = augmentation(
                img,
                bbox_orig,
                'eval',
                exclude_flip=True,
                input_img_shape=(256, 256),
                mask=alpha,
                base_scale=1.3,
                scale_factor=0.2,
                rot_factor=0,
                shift_wh=[bbox_orig[2], bbox_orig[3]],
                gaussian_std=3,
                bordervalue=bgcolor.tolist()
            )
        except Exception as e:
            print('[WARN] augmentation failed:', frame, e)
            continue

        img_save = os.path.join(out_img, f'{safe}.jpg')
        mask_save = os.path.join(out_mask, f'{safe}.png')

        if alpha_aug is not None:
            a = alpha_aug.astype(np.float32)
            if a.max() > 1.5:
                a /= 255.0
            a = np.clip(a, 0, 1)
            a3 = np.repeat(a[..., None], 3, axis=2)
            img_masked = (img_aug.astype(np.float32) * a3).astype(np.uint8)
            cv2.imwrite(img_save, img_masked)
            cv2.imwrite(mask_save, (a * 255).astype(np.uint8))
        else:
            cv2.imwrite(img_save, img_aug.astype(np.uint8))

        cam_sub = None
        if cam_orig is not None and 'intrinsics' in cam_orig:
            K = cam_orig['intrinsics'][:3, :3].copy()
            K[:2, 2] = trans_point2d(K[:2, 2], img2bb_trans)
            K[[0, 1], [0, 1]] *= 256 / (bbox_orig[2] * aug_param[1])

            if mesh_orig is not None:
                Rh = mesh_orig.get('Rh', np.zeros(3))
                Th = mesh_orig.get('Th', np.zeros(3))
            else:
                Rh = np.zeros(3)
                Th = np.zeros(3)

            E = apply_global_tfm_to_camera(np.eye(4), Rh=Rh, Th=Th)
            cam_sub = {
                'intrinsics': K,
                'extrinsics': E,
                'distortions': cam_orig.get('distortions', np.zeros(5))
            }

        bbox_sub = {frame: bbox_orig}
        img_type = 'rgb'

        with open(out_pkl, 'wb') as f:
            pickle.dump([cam_sub, mesh_orig, bbox_sub, img_type], f)

        written += 1

    print(f'[DONE] Generated {written} samples')

if __name__ == '__main__':
    main()
