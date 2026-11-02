# Selected experiment source and provenance

These files preserve the **server versions used for the recorded runs**. Their relative paths are unchanged because the scripts import one another through sibling experiment directories. Checkpoints, data, manifests, logs, and predictions are not included; the hard-coded server paths and hash guards must be satisfied before re-running elsewhere.

The recorded evaluator command points to [`runtime/octave_eval.sh`](runtime/octave_eval.sh), also archived. It invokes the original server's `/opt/anaconda3` `visdrone-eval` environment; a new host must supply an equivalent Octave installation.

| Run | Main source (SHA-256) | Locked protocol and command |
| --- | --- | --- |
| Matched raw/latent probe | [`run_matched_probe.py`](experiments/grid_k_budget_20261021/matched_probe_20261030/run_matched_probe.py) `16b58ccab4e690abc6a80d8f889a8c66a3ef747c2796db7f610b7759732043e3` | [`raw`](experiments/grid_k_budget_20261021/matched_probe_20261030/run/raw/protocol.json) and [`latent`](experiments/grid_k_budget_20261021/matched_probe_20261030/run/latent/protocol.json) protocols; each has a sibling `train_command.txt` |
| Six-checkpoint targeted evaluation | [`matched_probe_targeted.py`](experiments/grid_k_budget_20261021/matched_probe_20261030/matched_probe_targeted.py) `dedafe8b388e125da3129a9d69428d7c1a78546495186ae044d379d49011495b` | Evaluates epochs 5, 10, 15, 20, 25, 30 for each arm on val546 |
| Balanced high-K codec | [`run_high_k.py`](experiments/grid_k_budget_20261021/high_k_sweep_20261023/run_high_k.py) `d9d1c41ebfe098a0991f9b8637f3e83bc6d8929d71933fe0c4a287c865885145` | [formal `run_v2` protocol](experiments/grid_k_budget_20261021/high_k_sweep_20261023/run_v2/protocol.json), [train command](experiments/grid_k_budget_20261021/high_k_sweep_20261023/run_v2/train_command.txt), [eval command](experiments/grid_k_budget_20261021/high_k_sweep_20261023/run_v2/eval_command.txt) |
| Experiment A: frozen encoder, train original decoder | [`run_frozen_decoder.py`](experiments/grid_k_budget_20261021/frozen_encoder_decoder_20261027/run_frozen_decoder.py) `7f799096471fc27d433bd052a5b2ad68caddd2597db3612f6d5d4d9f15642cb3` | [`run` protocol](experiments/grid_k_budget_20261021/frozen_encoder_decoder_20261027/run/protocol.json), [train command](experiments/grid_k_budget_20261021/frozen_encoder_decoder_20261027/run/train_command.txt), [eval command](experiments/grid_k_budget_20261021/frozen_encoder_decoder_20261027/run/eval_command.txt) |

The matched probe defines the common per-level `256→128→256` convolution body, a latent-only learned `16→256` adapter, raw/latent input construction, detector hook, loss, checkpoint loading, and evaluation in `run_matched_probe.py`. `matched_probe_targeted.py` imports that file; it is the exact targeted-evaluation version named in the report. The direct import chain is also archived: `run_latent_probe.py` → `run_frozen_decoder.py` → `run_latent_direct.py` → `run_pg.py` → `run_high_k.py` → `run_extended.py`. These scripts are dependencies, not a claim that their other diagnostic results are archived here.

From the project root on the original server, the recorded commands were:

```sh
python3 experiments/grid_k_budget_20261021/high_k_sweep_20261023/run_high_k.py train
python3 experiments/grid_k_budget_20261021/high_k_sweep_20261023/run_high_k.py evaluate
python3 experiments/grid_k_budget_20261021/frozen_encoder_decoder_20261027/run_frozen_decoder.py train
python3 experiments/grid_k_budget_20261021/frozen_encoder_decoder_20261027/run_frozen_decoder.py evaluate
python3 experiments/grid_k_budget_20261021/matched_probe_20261030/run_matched_probe.py train --arm raw
python3 experiments/grid_k_budget_20261021/matched_probe_20261030/run_matched_probe.py train --arm latent
python3 experiments/grid_k_budget_20261021/matched_probe_20261030/matched_probe_targeted.py evaluate --arm raw
python3 experiments/grid_k_budget_20261021/matched_probe_20261030/matched_probe_targeted.py evaluate --arm latent
```

High-K `run_v2` uses seed 101, 30 epochs, a 128×256 grid, K=131072/262144/524288 complex symbols per image, and all three K values per training batch before one optimizer update. The source checks the latent and wire shapes, detector freeze, and mean complex symbol power. Its loss is detector loss plus `10 ×` valid-region relative feature MSE; AdamW uses LR `2e-4`, weight decay `1e-4`, batch size `2`, FP32. It loads the stage-B warmstart with SHA-256 `6b74d443714c5741f25c3263731c6809caae5c51c4d19f27e5f0f1f8170eaebb`. The model source is [`variable_grid_codec.py`](variable_grid_codec.py). `noise=false`; 10 dB is only metadata.

Experiment A starts from the high-K epoch-24 checkpoint (SHA-256 `140abdfd34af625090b3fea5671c63af195c1bf5fdbbdc50eabaa88787224604`), freezes the semantic encoder and detector, and trains the existing decoder for 30 epochs with the same optimizer and loss settings. It passes the latent directly; `N_symbol=0` and it is not a communication result. The matched raw/latent probe uses that same source checkpoint, seed 101, and no wire transmission. The locked train6467 and val546 manifest hashes are `840670f4b4f4b5f32117844978ad1213b3bee899f136fbe611116b5c48c512e0` and `3af48a18c687df3ebd41973731ded6f3eac1e2e43ccb965d3dc4bbb5fd7351c3`.
