# VisDrone2019-DET dataset archive

The original archives on the experiment server were:

| Archive | SHA-256 | Raw image count |
|---|---|---:|
| `VisDrone2019-DET-train.zip` | `86a77eba93137bfc16e4993860de9245b0675c0dba0d3ab98fb458699e256f84` | 6471 |
| `VisDrone2019-DET-val.zip` | `abeea063037e5d20398837deb11084e652402a34ddf4f207bdf541a6f2a35ef9` | 548 |
| `VisDrone2019-DET-test-dev.zip` | `b28a94b06dfd9e36ce77ff8155fb82b9d2c030f198a76105933bd56c6ea6a68d` | 1610 |

Use the official [VisDrone dataset release](https://github.com/VisDrone/VisDrone-Dataset) if the Git LFS files are absent. Retain the original dataset terms and citation. Extract to a common directory containing `VisDrone2019-DET-train/`, `VisDrone2019-DET-val/`, and `VisDrone2019-DET-test-dev/`, each with `images/` and `annotations/`.

`prepare_v2.py` excludes four train IDs and two val IDs because of duplicate/conflicting samples, giving train6467 and val546. The excluded IDs are recorded directly in `prepare_v2.py`; it regenerates split manifests with per-image and annotation hashes. The detector and codec evaluations described in this project used val546 after this lock. Keep test-dev out of model or K selection.

The dataset archives are deliberately not uploaded to this source archive. Obtain the original files from the official VisDrone release and verify each SHA-256 against this table before use.
