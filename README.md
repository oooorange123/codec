# VisDrone semantic feature codec archive

This repository archives the **training and inference source** for a frozen Deformable DETR detector and experimental feature codecs on VisDrone2019-DET. It preserves the original source implementations; it does not claim that the codec matched the detector's direct-feature accuracy or that SNR adaptation was validated.

## Included source

- `run_server.py`, `modeling.py`, `visdrone_data.py`, `codec.py`: detector fine-tuning, feature extraction, codec training, prediction export, and data loading.
- `fixed_grid_codec.py`, `prepare_v2.py`, `run_v2_stage_b.py`: the 64×64 variable-length stage-B codec and its locked-data training protocol.
- `variable_grid_codec.py`: the 128×256 conditional-K model definition. It maps K=131072/262144/524288 semantic scalars to the same number of complex channel uses per image.
- `experiments/grid_k_budget_20261021/`: the exact matched-probe, high-K training, and frozen-encoder decoder (experiment A) scripts requested for audit, their direct Python dependencies, and the locked run protocols and commands. See [EXPERIMENT_ARCHIVE.md](EXPERIMENT_ARCHIVE.md).
- `eval_official.m`, `eval_size_supplemental.m`, `vendor/VisDrone2018-DET-toolkit`: VisDrone evaluation code.
- `vendor/Deformable-DETR`: the detector source and CUDA attention operator, based on upstream commit `11169a60c33333af00a4849f1808023eba96a931`. Local source patches disable automatic ImageNet weight download and optionally enable a reference attention backend. See upstream `LICENSE`.
- `selftest.py`: basic model/wire shape and gradient sanity checks used by the main CLI.

Other performance-loss investigations, logs, checkpoints, predictions, cache files, compiled CUDA binaries, and the Python runtime environment remain excluded. The VisDrone evaluator is based on upstream `VisDrone2018-DET-toolkit` commit `005445782213e20cb91bc50a597db3dd949e749a` and is identified by its upstream research-use notice.

## Data and weights

See [DATASET.md](DATASET.md) for the three original VisDrone2019-DET archives, SHA-256 hashes, split counts, and the exact train/validation exclusions. The dataset archives are not included in this repository. Download the three official archives, verify the SHA-256 hashes in `DATASET.md`, and extract them below a common data root before running the code.

The trained detector and codec checkpoints and the 458 MB COCO pretrained detector checkpoint are **not** part of this source archive. Exact published AP numbers require the corresponding checkpoints. `prepare_v2.py` pins the detector checkpoint SHA-256 and regenerates machine-local manifests with image and annotation hashes.

## Environment and entry points

The code was used with Python 3.10, PyTorch, torchvision, CUDA, pycocotools, Pillow, NumPy, SciPy, and Octave/Matlab. Install the dependencies in `vendor/Deformable-DETR/requirements.txt` and build its CUDA operator with `bash vendor/Deformable-DETR/models/ops/make.sh` on a compatible GPU host. The compiled `.so` is intentionally excluded.

Run `python run_server.py --help` for the detector and original codec CLI. Train the detector with `python run_server.py train-detector --data-root DATA_ROOT --pretrained COCO_R50_CHECKPOINT --eval-bin octave --out OUTPUT_DIR` plus hardware and training options from `--help`. Once a compatible frozen detector checkpoint exists, `prepare_v2.py` locks the train/validation manifests, and `run_v2_stage_b.py --help` describes stage-B codec training. The later 128×256 high-K training command is recorded in [EXPERIMENT_ARCHIVE.md](EXPERIMENT_ARCHIVE.md).

The reference stage-B runs used a clean channel (`noise=false`); a 10 dB field in metadata did not add noise. Per-image geometry/shape/mask information was supplied at the receiver and was not counted in K. The wire tensor holds float I/Q samples; its in-memory byte size is not an over-the-air bit rate.
