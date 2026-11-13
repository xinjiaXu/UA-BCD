# UA-BCD Server Actions

Server checkout: `/home/xxj/code/DINOCLIP_comparative_test/UA-BCD`  
Dataset root: `/home/xxj/Data`

## Clone and update

First time:

```bash
cd /home/xxj/code/DINOCLIP_comparative_test
git clone --branch reproduce https://github.com/xinjiaXu/UA-BCD.git UA-BCD
```

Later updates:

```bash
cd /home/xxj/code/DINOCLIP_comparative_test/UA-BCD
git switch reproduce
git pull --ff-only
```

## Environment and pretrained backbone

Create the isolated environment and verify the five dataset layouts:

```bash
cd /home/xxj/code/DINOCLIP_comparative_test/UA-BCD
bash setup_server_env.sh
conda activate uabcd-rs
bash prepare_server_data.sh /home/xxj/Data
```

The model loads PVT-v2-B2 from `pretrained_model/pvt_v2_b2.pth`. The official README links the authors' Baidu Disk download (extraction code `abmg`). Download/extract that weight and place it at the exact path above before training. Checkpoints and logs are ignored by Git.

## Parameters and FLOPs

The profiler measures the UABCD inference network used by `test.py`; it excludes EUE, which is an auxiliary training-time module. Profiling uses batch 1 and a pair of 256x256 RGB images. It initializes the backbone randomly, so no PVT checkpoint is needed for profiling.

```bash
cd /home/xxj/code/DINOCLIP_comparative_test/UA-BCD
conda activate uabcd-rs
python -m pip install -r requirements_profile.txt
CUDA_VISIBLE_DEVICES=0 python profile_model.py --device cuda:0 \
  --output UA-BCD_work_dirs/profile_256.json
```

Use `Size(M)` and `Flops(G)` from the output. If `unsupported_ops` is non-empty, report the FLOPs as partial under fvcore's convention of one multiply-add counted as one FLOP.

## Dataset layout

The wrappers use the existing server folders directly; no dataset copy or generated list file is needed.

| Key | Dataset root | Image A | Image B | Label |
| --- | --- | --- | --- | --- |
| `levircd` | `/home/xxj/Data/LEVIR_CD` | `A` | `B` | `label` |
| `levircdplus` | `/home/xxj/Data/LEVIR_CD_PLUS/cut_data` | `A` | `B` | `label` |
| `cdd` | `/home/xxj/Data/CDD` | `A` | `B` | `OUT` |
| `whucd` | `/home/xxj/Data/WHUCD` | `A` | `B` | `label` |
| `sysucd` | `/home/xxj/Data/SYSU_CD` | `time1` | `time2` | `label` |

The adapted dataloader reads PNG/JPEG/TIFF image and mask files and expects binary masks encoded as black/white RGB values.

## Train

```bash
bash run_train.sh levircd
bash run_train.sh levircdplus
bash run_train.sh cdd
bash run_train.sh whucd
bash run_train.sh sysucd
```

The wrappers preserve the source defaults: 200 epochs, batch size 8, input size 256, latent dimension 8, 5 stochastic forward passes, Adam, both learning rates `1e-4`, and learning-rate decay 0.1 every 50 epochs. Training validates on the `val` split and saves the best-F1 segmentation model. Select GPU 1 with `GPU_ID=1`; set `NUM_WORKERS=4` or `BATCH_SIZE=8` to override runtime settings.

## Test

After training that dataset:

```bash
bash run_test.sh levircd
bash run_test.sh levircdplus
bash run_test.sh cdd
bash run_test.sh whucd
bash run_test.sh sysucd
```

Test metrics and prediction masks are written below `UA-BCD_work_dirs/<key>/test_predictions/`.
