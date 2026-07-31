# Mask2Former / full COCO 2017 instance campaign

This directory prepares a production three-mode AutoML campaign for the exact
TAO network identifier `mask2former` on the complete official COCO 2017
instance-segmentation train and validation splits.

It contains no CPU model execution, model smoke test, mini-step, synthetic
input benchmark, or SLURM submission. Every eventual model job uses the pinned
TAO 7.1 SQSH on one node with eight A100 GPUs.

## Scientific metric contract and qualification status

The task is COCO instance segmentation and the primary accuracy objective is
`segm_val_mAP` (COCO mask AP). Semantic `mIoU` is not an alias for mask AP and
is never accepted as one.

TAO PyTorch commit
`c2e86fe1646ebe89fc280083797dcc544ce88322` adds task-aware routing:
in-epoch validation reports `segm_val_mAP` and `segm_val_mAP50`, while
standalone evaluation reports the split-correct `segm_test_mAP` and
`segm_test_mAP50`. The campaign records the standalone names unchanged and
explicitly binds `segm_test_mAP` to its canonical `segm_val_mAP` accuracy
objective. It does not relabel TAO output or accept semantic mIoU.

The deterministic source overlay has SHA-256
`c395474592d557e0179066c1f99d5cb8f352e10e501621d57043782440dea8c2`
and is staged at:

```text
/lustre/fsw/portfolios/edgeai/users/rarunachalam/tao-pytorch-overlays/mask2former-instance-ap/c2e86fe1646ebe89fc280083797dcc544ce88322
```

The implementation blocker is fixed in source, but the model remains
fail-closed until that exact runtime is applied to the pinned SQSH, the direct
full-GPU qualification succeeds, and the PTM registry record is independently
promoted to `supported`.

## Frozen dataset

The prepared read-only Lustre root is:

```text
/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/cross_model_automl_20260729/coco2017_instance_panoptic_v1
```

The Mask2Former profile uses only the official instance contract:

- `images/train2017`: 118,287 images;
- `annotations/instances_train2017.json`: 860,001 annotations;
- `images/val2017`: 5,000 images;
- `annotations/instances_val2017.json`: 36,781 annotations;
- `tao/label_map_instance.json`: official 80 thing classes;
- `dataset.type: coco`, `dataset.contiguous_id: true`;
- `model.mode: instance`, `model.sem_seg_head.num_classes: 80`.

The immutable stage manifest SHA-256 is
`437ff12490637950707b9b951d820ea34d38b926080a478a5d182c2d284a0c5d`.
The 246,593-entry file manifest SHA-256 is
`10566a60498de9998154f44a34445a488c9f030e09f2a7346d20a4a1c55f804e`.
All remote hashes and file-set checks passed, and the staged files have zero
writable entries.

## PTM and objective-aware search

The repository-owned official registry currently contains exactly one
Mask2Former arm:

```text
mask2former.coco.swin_tiny.trainable.v1.0
```

It remains `unverified`; runtime does not mutate or bypass that status. Direct
full-dataset, three-epoch training and standalone evaluation must succeed on
one node/eight GPUs, emit `segm_val_mAP` and `segm_test_mAP` respectively,
produce mask AP above the frozen experiment sanity gate, and then the exact
registry record must be independently reviewed and marked `supported`.
Terminal failures are retained as exclusions.

Qualification is expected to precede the reviewed registry promotion. The
gate therefore permits the evidence envelope to name the earlier registry
digest, while binding the immutable NGC identity, observed checkpoint digest,
size, and workflow digest. After promotion it also requires the current
repository record to be `supported` and, when the promoted record carries a
checkpoint checksum, requires an exact checksum match. This avoids a
qualification/promotion sealing cycle without allowing evidence to bypass the
repository status.

Manifest sealing also requires the repository preflight/downloader to stage
the exact immutable NGC member on Lustre and write
`mask2former_coco2017_ptm_qualification_v1/ptm_stage_manifest.json`. Its
canonical content hash and raw file hash are both embedded in the campaign
contract. The record contains the registry SHA, stable PTM ID, immutable NGC
resource/version/member identity, observed checkpoint SHA and size, read-only
state, and explicit zero CPU/smoke/mini-step counters. A self-signed or
unbound stage file cannot authorize qualification.

PTM identity is represented as a hierarchical non-ordinal outer arm. The one
current arm is not encoded as an ordinal scalar. The common inner search is
frozen before any result:

```text
model.mask_former.num_object_queries: 50..200
model.mask_former.dec_layers: 4..10
dataset.augmentation.test_min_size: 480..800
train.optim.lr: 2e-5..5e-4
train.optim.weight_decay: 1e-4..0.10
```

The three campaigns are independent jobs with empty, separate observation
namespaces and 20 recommendations each:

| Mode | Recommendation acquisition | Final selection policy |
| --- | --- | --- |
| Accuracy | Expected improvement on mask AP | Highest valid mask AP |
| Latency | Constrained expected improvement | Raw-minimum-anchored equivalent-fastest cohort at 90% retained accuracy |
| Multi-objective | ParEGO expected improvement on mask AP and latency | Independent rank-zero normalized augmented-Chebyshev compromise |

Latency retention applies only to latency mode. Multi-objective mode does not
inherit it.

All intervention and selection-isolation flags are frozen `false`.
Recommendations, PTM arms, parameter values, seeds, budget, threshold, and
winners cannot be injected or overridden by an agent.

## Latency contract

Selection-time latency uses 16 immutable real COCO validation images, batch
size one, FP32, and model-forward scope. It excludes file I/O, preprocessing,
host-to-device transfer, instance postprocessing, serialization, and metric
accumulation. Every candidate uses:

- 50 warm-ups;
- five rounds of 100 timed requests;
- eight synchronized replicas;
- 4,000 raw samples;
- median, p95, MAD, IQR, robust CV, bootstrap interval, round drift, and
  device-spread quality gates.

## Automatic gating

After a sealed manifest exists, `--automatic-trigger --launch` waits for all
immutable data, SQSH, PTM evidence, and supported-registry gates. It does not
ask for confirmation. Once ready, the three independently seeded mode
controllers start together. Candidate zero in each mode performs the real
full train, standalone evaluation, and stabilized latency workflow. The
remaining 19 recommendations per mode are released automatically only after
all three candidate-zero workflows pass.

The automatic trigger currently remains blocked by the unexecuted full-GPU
runtime qualification and PTM-support requirements described above. No model
or scheduler job was launched while preparing this directory.

## Reproduction after prerequisites are reviewed

The dataset staging commit must first be integrated byte-for-byte so the
manifest generator can bind the recorded semantic SHA:

```bash
cd /localhome/local-rarunachalam/tao-automl

python -m \
  experiments.cross_model_automl_20260729.mask2former_coco2017_campaign.manifest_generator \
  --output \
  /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/mask2former_coco2017_three_mode/campaign.v1.json

# Inspect the direct-full qualification plan. Add --launch only after the
# task-correct mask-AP runtime and immutable PTM stage are independently ready.
python -m \
  experiments.cross_model_automl_20260729.mask2former_coco2017_campaign.qualification_campaign \
  --contract \
  /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/mask2former_coco2017_three_mode/campaign.v1.json

python -m \
  experiments.cross_model_automl_20260729.mask2former_coco2017_campaign.run_campaign \
  --contract \
  /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/mask2former_coco2017_three_mode/campaign.v1.json \
  --automatic-trigger \
  --launch
```

The command above is intentionally not runnable past the gate while the
repository registry remains unverified or the exact runtime fails to emit
`segm_val_mAP` for validation and `segm_test_mAP` for standalone evaluation.
