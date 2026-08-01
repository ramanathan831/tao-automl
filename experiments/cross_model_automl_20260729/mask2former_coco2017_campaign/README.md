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

The directory and every file are read-only. Every qualification, AutoML
training, standalone evaluation, and latency command checks the sealed
installer identity, verifies the archive, prepares a complete temporary
package mirror, and prepends it through `PYTHONPATH`. The package installed in
the pinned SQSH is never mutated. A changed archive, installer, source commit,
or writable remote stage fails before the TAO action starts.

The implementation blocker is fixed and bound into the campaign, but the
model remains fail-closed until direct full-GPU qualification succeeds and the
PTM registry record is independently promoted to `supported`.

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

The completed data-only stage contains the exact 569,716,712-byte NGC member:

```text
checkpoint SHA-256: 93f7e5a3ed960a9d6723b42e55e3cecc4aca9ef11bd5e96680bef2789fa3c356
manifest content SHA-256: 141d14f9b11e3cf81c087d7d05f4e054c885ced1be18554f9832e0cbc9b28bcc
manifest file SHA-256: 3d51ad23d237b8472ebff629dc9ceb7909123c462683f899f4eabb6f4cc3166e
remote mode: 0444
```

The staging operation performed zero CPU/GPU model runs and submitted zero
scheduler jobs.

## Qualification/runtime v3 slice-safe continuation

The frozen v1 qualification correctly used the SLURM skill's four-hour
allocation, 3.8-hour inner timeout, and automatic self-requeue. The direct
full-COCO run showed approximately 90 minutes per epoch and repeatedly reached
epoch 1. Its checkpoint interval was three epochs, however, so no checkpoint
was written before the 3.8-hour cutoff and every requeued execution restarted
at epoch 0. Runtime v2 attempted an eight-hour envelope, but `polar3` has a
four-hour maximum and the scheduler rejected that request before GPU work.

Qualification/runtime v3 retains the standard SLURM execution contract and
makes training resumable across its automatic slices:

```text
partition: polar3
SLURM allocation: 4.0 hours
SDK inner timeout: 3.8 hours
SLURM self-requeue: enabled
checkpoint interval: 1 epoch
resume candidates: exact regular non-symlink
  results_dir/train/model_epoch_<epoch>_step_<step>.pth
resume selection: numeric max (epoch, step), then filename
missing checkpoint: leave train.resume_training_checkpoint_path blank
selected checkpoint: inject train.resume_training_checkpoint_path and enable
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 for this trusted same-job artifact
training epochs: 3 (unchanged)
candidate budget per mode: 20 (unchanged)
search space, seeds, metrics, PTM, retry cap: unchanged
```

Every invocation writes
`mask2former_checkpoint_resume_decision.json` beside the generated spec. It
records the exact search directory, eligible count, selected epoch/step/path,
policy, and trust decision in an integrity-hashed decision record without
credentials. It does not rehash the large checkpoint on every slice. The same
wrapper is bound to direct qualification and every AutoML candidate training
command.

The v1 and v2 runtime trees remain immutable at
`mask2former_coco2017_ptm_qualification_v1` and
`mask2former_coco2017_ptm_qualification_v2`. The completed data-only v1 PTM
stage is reused by exact hash; no v1/v2 progress or incomplete evidence can
satisfy the v3 gate. New qualification evidence is written under
`mask2former_coco2017_ptm_qualification_v3`, and the later AutoML campaign
uses `mask2former_coco2017_three_mode_v3`. This is a checkpoint-continuation
correction, not a training-fidelity or search-policy change.

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

The automatic trigger remains blocked until the v3 full-GPU qualification
completes and the PTM-support requirements described above are satisfied.
The incomplete v1 and scheduler-rejected v2 qualifications are retained as
historical evidence and are not reused as successful gate results.

## Exact v3 qualification and automatic-launch commands

```bash
cd /localhome/local-rarunachalam/tao-automl

export PATH=/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1

# Seal v3 from a clean source commit. The immutable v1 PTM-stage manifest is
# consumed by hash and is not regenerated.
python -m \
  experiments.cross_model_automl_20260729.mask2former_coco2017_campaign.manifest_generator \
  --output \
  /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/mask2former_coco2017_three_mode_v3/campaign.v3.json

# This is plan-only and constructs no scheduler client.
python -m \
  experiments.cross_model_automl_20260729.mask2former_coco2017_campaign.qualification_campaign \
  --contract \
  /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/mask2former_coco2017_three_mode_v3/campaign.v3.json

# One direct full three-epoch train followed by standalone full validation,
# on one node/eight A100s in the pinned SQSH. No CPU/smoke/mini-step path.
python -m \
  experiments.cross_model_automl_20260729.mask2former_coco2017_campaign.qualification_campaign \
  --contract \
  /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/mask2former_coco2017_three_mode_v3/campaign.v3.json \
  --runtime-root \
  /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/mask2former_coco2017_ptm_qualification_v3 \
  --launch

# After successful qualification and independent registry promotion, reseal
# the same v3 contract path from the promoted clean source commit, then start
# the automatic three-mode trigger.
python -m \
  experiments.cross_model_automl_20260729.mask2former_coco2017_campaign.run_campaign \
  --contract \
  /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/mask2former_coco2017_three_mode_v3/campaign.v3.json \
  --runtime-root \
  /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/mask2former_coco2017_three_mode_v3 \
  --automatic-trigger \
  --launch
```

The command above is intentionally not runnable past the gate while the
repository registry remains unverified or the exact runtime fails to emit
`segm_val_mAP` for validation and `segm_test_mAP` for standalone evaluation.
