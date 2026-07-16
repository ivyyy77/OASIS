# OASIS

OASIS reconstructs and adapts an animatable 3D Gaussian hand avatar from a single image. This repository contains the training, InterHand adaptation, wild-image adaptation, editing, annotation generation, and inference code required by the OASIS pipeline.

## Installation

The release was validated on Linux with Python 3.10, PyTorch 2.12.0+cu126, and an NVIDIA GH200 GPU. Install a PyTorch build compatible with the CUDA toolkit on your machine before building the CUDA extensions.

```bash
git clone https://github.com/ivyyy77/oasis-hand-code.git
cd oasis-hand-code

conda create -n oasis python=3.10 -y
conda activate oasis

pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install "git+https://github.com/ashawkey/diff-gaussian-rasterization.git"
pip install ./third_party/simple-knn
```

`xformers` is optional. The DINOv2 encoder falls back to the standard PyTorch attention implementation when it is unavailable.

## Runtime assets and checkpoint

Small OASIS-specific MANO runtime files are included under `runtime_assets/`. The LHM prior assets are downloaded automatically on the first run when `pretrained_models/` does not exist. A prepared prior directory has the following runtime layout:

```text
pretrained_models/
  dense_sample_points/manohd_semantic.ply
  human_model_files/mano/
  mano_subdiv/mano_subdiv_2.pth
```

Download the released OASIS checkpoint and retain its iteration number in the filename:

```bash
mkdir -p checkpoint
gdown --fuzzy "https://drive.google.com/file/d/1UVMyPexOIp4GT0We31dW5kTJjEd7Wcdw/view?usp=drive_link" \
  -O checkpoint/iteration_30000.ckpt
```

Model weights, datasets, experiment outputs, and compiled CUDA binaries are intentionally excluded from Git.

## Data layout

Training uses InterHand2.6M at 5 fps with the HandAvatar/OHTA preprocessing layout:

```text
${INTERHAND_ROOT}/
  annotations/
    train/InterHand2.6M_train_data.json
    train/InterHand2.6M_train_camera.json
    train/InterHand2.6M_train_joint_3d.json
    train/InterHand2.6M_train_MANO_NeuralAnnot.json
    test/
  InterHand2.6M_5fps_batch1/
    images/
    masks_removeblack/
    preprocess_ohta_our_full/
```

Wild-image and editing inputs use sibling `images`, `masks`, and `anno` directories. Each image needs a foreground mask and a MANO annotation pickle with the same basename. Editing additionally uses `masks/<name>_edit.png`.

```text
sample/
  images/name.jpg
  masks/name.png
  masks/name_edit.png
  anno/name.pkl
```

The repository includes runnable examples under `example_data/interhand2.6m`, `example_data/editing`, and `example_data/text-to-avatar`.

To export a prepared InterHand frame into this format:

```bash
python generate_interhand_anno.py \
  --dataset-root "$INTERHAND_ROOT" \
  --output-root example_data/interhand2.6m \
  --subject test/Capture0/ROM03_RT_No_Occlusion \
  --phase test \
  --frame test/Capture0/ROM03_RT_No_Occlusion/cam400272/image15012.jpg
```

## One-shot adaptation and inference

The command below runs color inversion, pseudo-view generation, finetuning, and evaluation for the included InterHand image:

```bash
python finetune_wild_id2_ohta.py infer.hand_lrm model_name=LHM-1B \
  --input-dir example_data/interhand2.6m/images/test_Capture0_ROM03_RT_No_Occlusion_cam400272_image15012.jpg \
  --checkpoint-file checkpoint/iteration_30000.ckpt \
  --output-path output/finetune_wild \
  --iter=400 \
  --iter_inversion_stage1=100 \
  --iter_inversion_stage2=0 \
  --pseudo-views=8 \
  --animate_to_handavatar=False
```

Set `--iter=0` to run inversion and inference without the subsequent finetuning stage. Set both inversion counts and `--iter` to zero for a checkpoint-loading and dataset-construction smoke test.

## Training

Both the InterHand preprocessing root and the HandAvatar evaluation root are explicit inputs. They can also be supplied through `INTERHAND_ROOT` and `HANDAVATAR_ROOT`.

```bash
python train_interhand.py infer.hand_lrm model_name=LHM-1B \
  --dataset-path "$INTERHAND_ROOT" \
  --handavatar-path "$HANDAVATAR_ROOT" \
  --checkpoint-path checkpoint/interhand \
  --output-path output/interhand \
  --iter=40000
```

Evaluate a released checkpoint with the same prepared datasets:

```bash
python train_interhand.py infer.hand_lrm model_name=LHM-1B \
  --dataset-path "$INTERHAND_ROOT" \
  --handavatar-path "$HANDAVATAR_ROOT" \
  --checkpoint-file checkpoint/iteration_30000.ckpt \
  --output-path output/eval_interhand \
  --only_eval
```

## InterHand adaptation

```bash
python finetune_interhand_ohta.py infer.hand_lrm model_name=LHM-1B \
  --input-dir example_data/interhand2.6m/images/test_Capture0_ROM03_RT_No_Occlusion_cam400272_image15012.jpg \
  --handavatar-path "$HANDAVATAR_ROOT" \
  --checkpoint-file checkpoint/iteration_30000.ckpt \
  --output-path output/finetune_interhand \
  --iter=1200 \
  --iter_inversion_stage1=100 \
  --iter_inversion_stage2=0 \
  --pseudo-views=8 \
  --animate_to_handavatar=False
```

## Editing

```bash
python finetune_edit_wild_ohta.py infer.hand_lrm model_name=LHM-1B \
  --input-dir example_data/editing/images/rose.jpg \
  --checkpoint-file checkpoint/iteration_30000.ckpt \
  --output-path output/finetune_edit \
  --iter=800 \
  --iter_inversion=100 \
  --edit-unmask-iter=0 \
  --edit-mask-weight=30 \
  --pseudo-views=8 \
  --animate_to_handavatar=False
```

## License

The code is released under the Apache License 2.0. MANO, InterHand2.6M, LHM, HandAvatar, OHTA, and other third-party components remain subject to their respective licenses.
