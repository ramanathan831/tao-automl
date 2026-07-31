# OneFormer / full COCO2017 campaign

This directory prepares three independent objective-aware AutoML jobs for the
exact TAO identifier `oneformer` on the complete official COCO2017
train/validation image and panoptic-annotation releases. Preparation performs
no CPU model run, model smoke, mini-step, GPU model run, or SLURM submission.

| Mode | Recommendation target | Final policy |
| --- | --- | --- |
| Accuracy | Expected improvement on `mIoU` | Highest valid accuracy |
| Latency | Constrained expected improvement | Raw-minimum-anchored equivalent-fastest cohort at 90% retained accuracy |
| Multi-objective | ParEGO expected improvement on mIoU and latency | Independent rank-zero augmented-Chebyshev compromise |

PTM identity is a hierarchical nonordinal arm. The four official NGC
checkpoints remain `unverified`; none is manually selected. Each arm must
complete one real full-COCO epoch plus standalone validation on one node/eight
A100s, and its exact registry record must then be independently promoted to
`supported`. Terminal failures remain preserved exclusions. The automatic
trigger waits until at least one arm satisfies both gates.

The read-only [static SQSH audit](static_sqsh_audit.v1.json) found blockers
before any model execution: the packaged train entrypoint calls an undefined
full-checkpoint loader, the implementation does not emit PQ, and the status
KPI consumed by the campaign is not based on a globally reduced distributed
confusion statistic. Consequently the four PTMs remain `unverified`, the
automatic trigger remains closed, and submitting qualification or AutoML jobs
with this pinned image would be invalid.

All campaign children use the pinned TAO 7.1 SQSH, one node/eight A100s, and
the native 133-category panoptic label map. Candidate zero runs independently
in all three modes. Only after all three first candidates pass training,
standalone validation, stabilized latency, recommendation-audit, and provenance
gates does the controller automatically release the remaining budget.

## Metric boundary

The current OneFormer train/evaluate implementation emits semantic `mIoU` from
native panoptic annotations. It does not emit Panoptic Quality (PQ). This
campaign records that exact metric and never relabels it as PQ. Consequently:

- it can validate objective-aware search and latency tradeoffs for the emitted
  OneFormer semantic-quality endpoint;
- it cannot establish the product claim that OneFormer panoptic PQ is
  optimized;
- a task-correct PQ path remains a separate implementation blocker.

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

## Seal and launch later

After the static SQSH blockers are fixed, direct full-run qualification
evidence exists, and registry support is reviewed:

```bash
cd /localhome/local-rarunachalam/tao-automl
python -m experiments.cross_model_automl_20260729.oneformer_coco2017_campaign.manifest_generator \
  --output /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/oneformer_coco2017_three_mode/campaign.v1.json

python -m experiments.cross_model_automl_20260729.oneformer_coco2017_campaign.run_campaign \
  --contract /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/oneformer_coco2017_three_mode/campaign.v1.json \
  --runtime-root /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/oneformer_coco2017_three_mode \
  --automatic-trigger \
  --launch
```

No post-gate confirmation is required. The automatic trigger itself performs
the transition once immutable prerequisites pass.
