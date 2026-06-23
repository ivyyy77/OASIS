# GitHub 正式发布整理计划

本仓库当前只是 private staging，不是正式开源版本。后续公开前，README 和仓库结构可按以下方式整理。

## 1. README 顶部

- 标题：OASIS / MOVA hand avatar reconstruction。
- 作者、机构、论文状态。
- 项目页、论文、代码、demo、视频链接。
- 一句话说明输入、输出和核心能力。

## 2. Updates

- 记录代码、模型、数据处理脚本、demo、论文版本的发布日期。
- 标明哪些功能已经发布，哪些仍在整理。

## 3. Installation

- Python、CUDA、PyTorch、3D Gaussian Splatting 相关依赖版本。
- 推荐 conda 环境创建命令。
- 可选依赖和常见安装问题。

## 4. Required Assets / MANO / SMPL-X / Hand Priors

- MANO、SMPL-X、hand priors、template mesh、UV assets 的下载位置或申请方式。
- 明确哪些文件不能随仓库直接分发。
- 提供目录结构示例，但不上传真实权重或受限资产。

## 5. Data Preparation

- 单图输入、mask、camera、InterHand/HanCo/自采数据的准备格式。
- 数据目录示例。
- 数据预处理脚本入口和参数说明。

## 6. One-shot Hand Avatar Creation

- 从单张图片创建 hand avatar 的最小命令。
- checkpoint、输出路径、渲染视角、导出 mesh / video 的参数。
- 常见失败案例与排查方式。

## 7. Evaluation

- PSNR、SSIM、LPIPS、identity / hand-specific metrics 的评测入口。
- benchmark 数据准备方式。
- 复现实验表格的命令模板。

## 8. Training

- 多身份 hand prior training。
- 单身份 one-shot adaptation / finetuning。
- 分布式训练、resume、checkpoint 保存策略。

## 9. Applications

- Texture editing。
- Text-to-avatar。
- Novel-view rendering。
- Animation / motion transfer。

## 10. Citation

- 论文 BibTeX。
- 相关依赖项目引用。

## 11. License

- 代码许可证。
- 预训练模型、第三方资产和数据集的单独许可证说明。

## 12. Acknowledgements

- 感谢使用或参考的项目、数据集、模型和工具。
- 明确第三方代码归属，不复制第三方 README 文本。

## 13. Repo Structure

- `LHM/`：核心模型、runner、rendering、utils。
- `scene/`：Gaussian / deformation 相关场景模块。
- `tools_utils/`：hand avatar、SMPL-X、metrics 和通用工具。
- `engine/`：分割、姿态估计等推理支撑模块。
- 根目录脚本：training、finetuning、testing、rendering 入口。

## 14. 公开前必须删除或替换的内容

- 所有 `logs/`、`output/`、`outputs/`、`results/`、`runs/`、`wandb/`、`exp/`、`experiments/`。
- 所有 `assets/`、`data/`、`datasets/`、`example_data/` 中的真实样例、图片、视频或隐私数据。
- 所有 `checkpoint/`、`checkpoints/`、`pretrained/`、`pretrained_models/`、`weights/`。
- 所有 `.pth`、`.pt`、`.ckpt`、`.tar`、`.tar.gz`、`.zip`、`.npz`、`.npy`、`.pkl`、`.jsonl`、视频、图片、mesh、binary 文件。
- 本地绝对路径、账号路径、机器路径、W&B key、token、secret、API key。
- 调试专用脚本、过期备份文件、`*_ori.py`、`*copy*`、临时实验脚本。
