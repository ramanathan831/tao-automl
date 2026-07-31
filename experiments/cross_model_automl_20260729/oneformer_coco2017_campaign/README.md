# OneFormer / full COCO2017 campaign

This directory prepares three independent objective-aware AutoML jobs for the
exact TAO identifier `oneformer` on the complete official COCO2017
train/validation image and panoptic-annotation releases. Preparation performs
no CPU model run, model smoke, mini-step, GPU model run, or SLURM submission.

| Mode | Recommendation target | Final policy |
| --- | --- | --- |
| Accuracy | Expected improvement on `PQ` | Highest valid accuracy |
| Latency | Constrained expected improvement | Raw-minimum-anchored equivalent-fastest cohort at 90% retained accuracy |
| Multi-objective | ParEGO expected improvement on PQ and latency | Independent rank-zero augmented-Chebyshev compromise |

PTM identity is a hierarchical nonordinal arm. The four official NGC
checkpoints remain `unverified`; none is manually selected. Each arm must
complete one real full-COCO epoch plus standalone validation on one node/eight
A100s, and its exact registry record must then be independently promoted to
`supported`. Terminal failures remain preserved exclusions. The automatic
trigger waits until at least one arm satisfies both gates.

The read-only [static SQSH audit](static_sqsh_audit.v1.json) records three
defects in the immutable base image: no full-checkpoint loader, no panoptic PQ
endpoint, and no globally reduced status metric. Those findings are preserved
unchanged. The campaign remediates them with the reviewed TAO PyTorch source
overlay at commit `c25a20e0d6e2cf98ccb80c16eb0d4d30bb40f600`, archive SHA-256
`6b976090fb264b319ba23e7092445f261fd1b445964400d3f879c2746247a4f3`.
Every training, standalone-evaluation, and latency command verifies and
installs that overlay before importing TAO PyTorch, and persists an installer
receipt. A missing, changed, or inapplicable overlay leaves the automatic
trigger closed.

All campaign children use the pinned TAO 7.1 SQSH, one node/eight A100s, and
the native 133-category panoptic label map. Candidate zero runs independently
in all three modes. Only after all three first candidates pass training,
standalone validation, stabilized latency, recommendation-audit, and provenance
gates does the controller automatically release the remaining budget.

## Metric contract

The campaign sets `evaluate.task: panoptic` in both training and standalone
evaluation specs. Its canonical objective is unit-scale `PQ`; standalone
evaluation accepts `test_PQ` (or the unprefixed `PQ` status key) and records it
as `PQ` for AutoML. It never substitutes semantic mIoU. The overlay computes
COCO-style PQ from native panoptic IDs and globally sums additive sufficient
statistics before deriving PQ/SQ/RQ, so every rank observes the same metric
and only global rank zero writes status.

## Frozen data and runtime

Dataset root:

```text
/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/cross_model_automl_20260729/coco2017_instance_panoptic_v1
```

The stage record SHA-256 is
`437ff12490637950707b9b951d820ea34d38b926080a478a5d182c2d284a0c5d`.
All 246,593 manifest entries passed remote verification and the frozen dataset
contains zero writable files.

Latency uses the same 16 raw validation images for every candidate, 50
warm-ups, five rounds of 100 synchronized model-forward samples on each of
eight replicas, and 4,000 samples per candidate. Preprocessing and
postprocessing are excluded; candidate-controlled test resolution is therefore
part of the measured inference graph input contract.

## Data-only PTM stage

The PTM stage resolves exactly the four official OneFormer records frozen by
the repository registry. It uses the production authenticated NGC HTTPS client
and atomic verified cache, then create-or-verifies immutable checkpoint bytes
and a read-only manifest. It imports no TAO model implementation, constructs no
scheduler client, and submits no job.

The physical publication root and canonical runtime root are deliberately
separate. This supports a login host where remote Lustre is mounted over SSHFS:
bytes are written and verified through the physical mount, while the manifest
contains only the canonical `/lustre/...` paths seen by cluster jobs.

```bash
cd /localhome/local-rarunachalam/tao-automl
python -m experiments.cross_model_automl_20260729.oneformer_coco2017_campaign.ptm_stage \
  --stage \
  --env-file /localhome/local-rarunachalam/.tao/config.env \
  --physical-publication-root /path/to/sshfs/mount/oneformer_v1 \
  --canonical-publication-root /lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam/ptms/cross_model_automl_20260729/oneformer_v1
```

Revalidation is network-free and uses the same explicit mapping:

```bash
python -m experiments.cross_model_automl_20260729.oneformer_coco2017_campaign.ptm_stage \
  --check-stage \
  --physical-publication-root /path/to/sshfs/mount/oneformer_v1 \
  --canonical-publication-root /lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam/ptms/cross_model_automl_20260729/oneformer_v1
```

An existing destination is reused only when its size, SHA-256, and read-only
mode are exact. Changed or writable bytes, unexpected files, registry drift,
and manifest drift are terminal errors; the stager never overwrites them.

## Seal and launch later

After the exact overlay is staged at its preregistered Lustre path, direct
full-run PQ qualification evidence exists, and registry support is reviewed:

```bash
cd /localhome/local-rarunachalam/tao-automl
python -m experiments.cross_model_automl_20260729.oneformer_coco2017_campaign.manifest_generator \
  --runtime-overlay /localhome/local-rarunachalam/.tao/artifacts/oneformer-runtime-product-fixes-c25a20e0/oneformer-runtime-overlay.tar \
  --output /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/oneformer_coco2017_three_mode/campaign.v1.json

python -m experiments.cross_model_automl_20260729.oneformer_coco2017_campaign.run_campaign \
  --contract /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/oneformer_coco2017_three_mode/campaign.v1.json \
  --runtime-root /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/oneformer_coco2017_three_mode \
  --automatic-trigger \
  --launch
```

No post-gate confirmation is required. The automatic trigger itself performs
the transition once immutable prerequisites pass. This preparation path does
not run a CPU model, model smoke, mini-step, GPU model, or SLURM job.
