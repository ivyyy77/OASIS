
"""Generate InterHand annotation pickle (anno_cam.pkl) used by the dataset.

Usage:
  python scripts/generate_interhand_anno.py --dataset-root /path/to/dataset --subject train/Capture0 --phase train --force
"""
import os
import argparse
import pickle
import re

from tools_utils.model.ohta.configs import cfg

import cv2
import numpy as np
from data.utils.augm_util import augmentation, trans_point2d
from data.utils.camera_util import apply_global_tfm_to_camera





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', default=os.environ.get('INTERHAND_ROOT'), help='Root path that contains InterHand annotations and images')
    parser.add_argument('--output-root', default='./example_data/interhand2.6m', help='Output directory for generated annotations, images, and masks')
    parser.add_argument('--subject', default='test/Capture0/ROM03_RT_No_Occlusion', help='Subject folder, e.g. train/Capture0')

    parser.add_argument('--phase', default='test', help='train or test or infer')
    parser.add_argument('--maxframes', type=int, default=-1)

    parser.add_argument('--frame', default=None, help='Optional frame name or full image path')



    args = parser.parse_args()
    if not args.dataset_root:
        parser.error('--dataset-root is required when INTERHAND_ROOT is not set')

    dataset_root = args.dataset_root
    subject = args.subject
    phase = args.phase


    image_dir = os.path.join(dataset_root, f'InterHand2.6M_{cfg.interhand.fps}fps_batch1/images')
    anno_name = os.path.join(image_dir.replace('images', 'preprocess'), subject, 'anno_cam.pkl')

    cameras = {}
    mesh_infos = {}
    bbox = {}
    framelist = []

    if os.path.exists(anno_name):
        print(f'Found existing combined annotation: {anno_name} -- will use it to generate per-frame pkls (won\'t overwrite existing per-frame pkls).')
        try:
            with open(anno_name, 'rb') as f:
                cameras, mesh_infos, bbox, framelist = pickle.load(f)
        except Exception as e:
            print('Failed to load existing anno pickle:', e)
            cameras = {}
            mesh_infos = {}
            bbox = {}
            framelist = []
    else:

        try:
            print('Combined anno not found; importing Dataset to trigger preprocessing (may require proper env).')
            from data.interhand.train import HandAvatarDataset as Dataset
            ds = Dataset(dataset_root, data_type='progress')
            cameras = getattr(ds, 'cameras', {})
            mesh_infos = getattr(ds, 'mesh_infos', {})
            bbox = getattr(ds, 'bbox', {})
            framelist = getattr(ds, 'framelist', [])
        except Exception as e:
            print('Could not import or run Dataset.preprocess. Reason:', e)
            print('You can either run the original preprocessing in a proper environment to create', anno_name)
            print('or provide an existing anno pickle. Exiting.')
            return


    if args.frame:
        frame_input = args.frame
        if os.path.exists(frame_input):
            if image_dir in frame_input:
                rel = frame_input.split(image_dir + os.sep, 1)[1]
            else:
                m = re.search(r'(train|test)/.+?\.jpg', frame_input)
                rel = m.group(0) if m else os.path.basename(frame_input)
        else:
            rel = frame_input

        if framelist and rel not in framelist:

            jpg_idx = rel.find('.jpg')
            if jpg_idx != -1:
                rel_trunc = rel[:jpg_idx+4]
            else:
                rel_trunc = rel

            if rel_trunc in framelist:
                rel = rel_trunc
            else:

                name_no_ext = os.path.splitext(os.path.basename(rel))[0]
                candidates = [f for f in framelist if name_no_ext in f]
                if candidates:
                    print(f"Note: using closest framelist match {candidates[0]} for requested {rel}")
                    rel = candidates[0]

        frames_to_process = [rel]
    else:
        frames_to_process = list(framelist)

    base_out = args.output_root
    out_dir = os.path.join(base_out, 'anno')
    out_img_dir = os.path.join(base_out, 'images')
    out_mask_dir = os.path.join(base_out, 'masks')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    written = 0
    for frame in frames_to_process:
        safe = frame.replace('/', '_')
        safe = safe.replace('.jpg', '')
        out_path = os.path.join(out_dir, f'{safe}.pkl')





        cam_orig = cameras.get(frame, None)
        mesh_orig = mesh_infos.get(frame, None)
        bbox_orig = bbox.get(frame, None)
        img_type = 'rgb'


        img_path = os.path.join(image_dir, frame)
        mask_path = img_path.replace('/images/', '/masks_removeblack/').replace('.jpg', '.png')
        if not os.path.exists(img_path):
            print('Image not found, skipping:', img_path)
            continue
        img = cv2.imread(img_path)
        if img is None:
            print('Failed read image, skipping:', img_path)
            continue

        alpha = None
        if os.path.exists(mask_path):
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                alpha = m


        bgcolor = np.array([0.0, 0.0, 0.0], dtype='float32')


        if bbox_orig is None:
            print('No bbox for frame, skipping:', frame)
            continue
        try:
            img_aug, img2bb_trans, bb2img_trans, aug_param, do_flip, scale_factor, alpha_aug = augmentation(
                img, bbox_orig, 'eval', exclude_flip=True, input_img_shape=(256, 256), mask=alpha,
                base_scale=1.3, scale_factor=0.2, rot_factor=0, shift_wh=[bbox_orig[2], bbox_orig[3]], gaussian_std=3, bordervalue=bgcolor.tolist())
        except Exception as e:
            print('augmentation failed for', frame, e)
            continue


        img_save_path = os.path.join(out_img_dir, f'{safe}.jpg')
        mask_save_path = os.path.join(out_mask_dir, f'{safe}.png')

        if alpha_aug is not None:
            a = alpha_aug.astype(np.float32)
            if a.max() > 1.5:
                a = a / 255.0
            a = np.clip(a, 0.0, 1.0)
            if a.ndim == 2:
                a3 = np.repeat(a[..., None], 3, axis=2)
            else:
                a3 = a
            img_masked = (img_aug.astype(np.float32) * a3).astype(np.uint8)
            cv2.imwrite(img_save_path, img_masked)

            mask_u8 = (a * 255).astype(np.uint8)
            cv2.imwrite(mask_save_path, mask_u8)
        else:
            cv2.imwrite(img_save_path, img_aug.astype(np.uint8))


        cam_sub = None
        if cam_orig is not None and 'intrinsics' in cam_orig:
            K = cam_orig['intrinsics'][:3, :3].copy()
            K[:2, 2] = trans_point2d(K[:2, 2], img2bb_trans)

            K[[0, 1], [0, 1]] = K[[0, 1], [0, 1]] * 256 / (bbox_orig[2] * aug_param[1])

            if mesh_orig is not None:
                Rh = mesh_orig.get('Rh', np.zeros(3))
                Th = mesh_orig.get('Th', np.zeros(3))
            else:
                Rh = np.zeros(3)
                Th = np.zeros(3)
            E_global = apply_global_tfm_to_camera(np.eye(4), Rh=Rh, Th=Th)
            E_global = np.eye(4)
            R = E_global[:3, :3]
            T = E_global[:3, 3]

            cam_sub = {'intrinsics': K, 'extrinsics': E_global, 'distortions': cam_orig.get('distortions', np.zeros(5) if cam_orig else np.zeros(5))}
        else:
            cam_sub = {frame: None}


        mesh_sub = mesh_orig
        bbox_sub = {frame: bbox_orig}

        with open(out_path, 'wb') as fo:
            pickle.dump([cam_sub, mesh_sub, bbox_sub, img_type], fo)
        written += 1
        print('Wrote:', out_path)

    print(f'Done. Wrote {written} per-frame pkl files to {out_dir}')



if __name__ == '__main__':
    main()
