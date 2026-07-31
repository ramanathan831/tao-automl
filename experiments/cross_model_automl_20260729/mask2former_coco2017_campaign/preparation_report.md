# Mask2Former COCO 2017 campaign preparation report

## Verdict

The production campaign is prepared but intentionally not launch-ready.

No CPU/model smoke, mini-step, local model execution, GPU model execution, or
SLURM submission was performed while preparing it. The launch gate fails
closed on two unresolved prerequisites:

1. the current TAO Mask2Former validation and test path does not emit
   task-correct COCO mask AP;
2. the one official Mask2Former PTM is still `unverified` and has not completed
   the direct full-run qualification/promotion sequence.

## Frozen scientific scope

| Field | Frozen value |
| --- | --- |
| TAO model identifier | `mask2former` |
| Task | COCO instance segmentation |
| Primary accuracy metric | `segm_val_mAP` |
| Dataset | Complete official COCO 2017 train/validation |
| Classes | Official 80-class instance label map |
| Dataset root | `/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/cross_model_automl_20260729/coco2017_instance_panoptic_v1` |
| Dataset stage SHA-256 | `437ff12490637950707b9b951d820ea34d38b926080a478a5d182c2d284a0c5d` |
| File-manifest SHA-256 | `10566a60498de9998154f44a34445a488c9f030e09f2a7346d20a4a1c55f804e` |
| Runtime | Pinned TAO 7.1 SQSH |
| Resources per model job | One node, eight A100 GPUs |
| Candidate budget | 20 per independent mode |
| Training fidelity | Three complete epochs per candidate |

The data gate verifies all 118,287 training images, 5,000 validation images,
860,001 training instance annotations, 36,781 validation instance
annotations, the instance JSON and label-map hashes, the 246,593-entry file
set, the byte-identical Lustre stage record, and zero remote writable entries.

## Root-cause evidence for the metric blocker

The inspected implementation at
`nvidia_tao_pytorch/cv/mask2former/model/pl_model.py` routes validation and
test through semantic inference. `val_epoch_end()` computes and publishes
`mIoU` and `ACC_all`; it does not run a COCO instance evaluator or publish
mask AP. `model.mode: instance` controls inference postprocessing but does not
change that validation metric path.

The repository-owned policy in `src/tao_automl/metric_sanity.py` already
records `mask2former` / `instance_segmentation` / `segm_val_mAP` as blocked for
this reason. The campaign preserves that decision. It does not reinterpret
the observed mIoU as mask AP.

## Objective-aware jobs

The three jobs have equal budgets and fidelity but separate observation
namespaces:

| Mode | Acquisition | Constraint | Final policy |
| --- | --- | --- | --- |
| Accuracy | Expected improvement | None | Highest valid mask AP |
| Latency | Constrained expected improvement | 90% of the job's best observed/final accuracy reference | Highest-accuracy member of the raw-minimum-anchored equivalent-fastest cohort |
| Multi-objective | ParEGO expected improvement | No inherited latency floor | Rank-zero normalized augmented-Chebyshev compromise |

The frozen inner search covers query count, decoder depth, evaluation input
resolution, learning rate, and weight decay. PTM identity is a hierarchical,
non-ordinal outer arm. The official inventory currently has one arm:
`mask2former.coco.swin_tiny.trainable.v1.0`.

All agent-intervention and validation-to-selection feedback flags are false.
Failed recommendations and direct qualification failures are terminal,
preserved records; replacement candidates are not injected.

## Latency protocol

Every candidate uses 16 immutable real validation images, FP32, batch size
one, 50 warm-ups, five rounds of 100 requests, and eight synchronized replicas
for 4,000 samples. The timed scope is Mask2Former model forward. I/O,
preprocessing, transfer, instance postprocessing, serialization, metric
accumulation, and distributed gather are excluded consistently. Median, p95,
MAD, IQR, robust CV, bootstrap interval, round drift, and device-spread gates
must pass.

## Automatic release behavior

The automatic trigger performs no work while data, PTM, registry, source,
wheel, SDK, skills, SQSH, and task-metric gates are incomplete. Once all are
valid, it starts the three independent controllers. It automatically releases
the remaining 19 recommendations per mode only after candidate zero in all
three modes passes full training, standalone validation, stabilized latency,
audit, and provenance checks.

## Required next sequence

1. Add and review a task-correct COCO instance mask-AP evaluator in the TAO
   Mask2Former validation and standalone evaluation path.
2. Stage the exact official NGC PTM on Lustre using the repository preflight
   downloader; freeze its observed digest, size, immutable source identity,
   and read-only stage manifest.
3. Seal the pre-promotion direct-full qualification contract.
4. Run the one real three-epoch, one-node/eight-A100 qualification workflow
   and retain success or terminal failure unchanged.
5. If and only if it succeeds, independently promote the exact registry
   record to `supported`, binding the evidence and observed checkpoint digest.
6. Seal the final AutoML campaign on the promoted source and start the
   automatic trigger.

No additional model or dataset is implicated by these blockers.

## Verification performed

The campaign-specific suite passed:

```text
26 passed
```

The combined production objective-acquisition, selection, PTM,
recommendation-audit, runtime, wheel, and campaign suites passed:

```text
553 passed
```

Only the established sklearn Gaussian-process convergence warnings were
observed. Python compilation and `git diff --check` also passed.
