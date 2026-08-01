# Mask2Former COCO 2017 campaign preparation report

## Verdict

The production campaign and task-correct runtime fix are prepared, but the
campaign is intentionally not launch-ready until direct GPU qualification.

No CPU/model smoke, mini-step, local model execution, GPU model execution, or
SLURM submission was performed while preparing it. TAO PyTorch commit
`c2e86fe1646ebe89fc280083797dcc544ce88322` now emits `segm_val_mAP` during
validation and the split-correct `segm_test_mAP` during standalone evaluation.
The deterministic source overlay is staged on Lustre with SHA-256
`c395474592d557e0179066c1f99d5cb8f352e10e501621d57043782440dea8c2`.
The production campaign now binds source commit
`c2e86fe1646ebe89fc280083797dcc544ce88322`, archive and installer byte
identities, and read-only remote state. Every Mask2Former action prepares the
overlay as a temporary package mirror and injects it through `PYTHONPATH`;
the package installed in the SQSH is not modified.

The exact official 569,716,712-byte Mask2Former Swin-T checkpoint is also
staged read-only on Lustre. Its observed SHA-256 is
`93f7e5a3ed960a9d6723b42e55e3cecc4aca9ef11bd5e96680bef2789fa3c356`;
the stage content SHA-256 is
`141d14f9b11e3cf81c087d7d05f4e054c885ced1be18554f9832e0cbc9b28bcc`
and its raw manifest-file SHA-256 is
`3d51ad23d237b8472ebff629dc9ceb7909123c462683f899f4eabb6f4cc3166e`.

The launch gate still fails closed because the exact runtime has not completed
the direct full-GPU qualification and the one official Mask2Former PTM remains
`unverified` pending that qualification and independent registry promotion.

## 2026-08-01 qualification/runtime v3 amendment

The first direct GPU qualification exposed a checkpoint-continuation defect,
not a need for a longer partition. The standard four-hour allocation and
3.8-hour inner timeout automatically requeued as designed, and each slice
reached epoch 1 at approximately 90 minutes per epoch. The frozen interval of
three epochs wrote no checkpoint before the cutoff, however, so each
execution restarted at epoch 0. V2's attempted eight-hour request was rejected
because `polar3` has a four-hour maximum; it performed no GPU work.

V3 restores the SLURM skill defaults of four hours and 3.8 hours with
self-requeue enabled. It changes the checkpoint interval to one epoch and
wraps every qualification and AutoML training command with deterministic
same-job continuation. Before training, the wrapper scans only
`results_dir/train`, ignores symlinks, malformed names, and unrelated paths,
selects the maximum numeric epoch/step from an exact
`model_epoch_<epoch>_step_<step>.pth` filename, injects
`train.resume_training_checkpoint_path`, and enables trusted loading only
for that own-job artifact. With no eligible checkpoint it explicitly leaves
the resume field blank. An integrity-hashed decision record is written beside
the generated spec on every execution.

Training remains three complete epochs; the search space, 20-candidate budget,
seeds, PTM, metrics, mode policies, and retry cap are unchanged. V1 and v2
runtime evidence remain untouched. V3 reuses only the immutable,
content-addressed v1 PTM stage and writes all new qualification/runtime state
to separate `*_v3` roots.

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

## Metric-routing resolution

The prior root cause was task-blind validation/test routing in
`nvidia_tao_pytorch/cv/mask2former/model/pl_model.py`. The runtime fix uses the
distributed COCO segmentation evaluator for `model.mode: instance`, preserves
semantic mIoU only for semantic mode, and labels panoptic semantic output as a
diagnostic rather than mask AP or PQ.

Training validation emits `segm_val_mAP` and `segm_val_mAP50`. Standalone
evaluation emits `segm_test_mAP` and `segm_test_mAP50`. The campaign preserves
those raw names and records an explicit `segm_test_mAP` to canonical
`segm_val_mAP` objective binding. It never reinterprets semantic mIoU.

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

1. Review the task-correct TAO PyTorch commit and sealed PYTHONPATH overlay
   contract.
2. Seal the pre-promotion direct-full qualification contract using the
   completed read-only PTM stage.
3. Run the one real three-epoch, one-node/eight-A100 qualification workflow
   and retain success or terminal failure unchanged.
4. If and only if it succeeds, independently promote the exact registry
   record to `supported`, binding the evidence and observed checkpoint digest.
5. Seal the final AutoML campaign on the promoted source and start the
   automatic trigger.

No additional model or dataset is implicated by these blockers.

## Verification performed

The prior campaign-specific suite passed after the v2 amendment:

```text
37 passed
```

The complete repository suite passed:

```text
970 passed, 1 skipped
```

The complete cross-model experiment suite passed:

```text
420 passed
```

Only three established sklearn Gaussian-process convergence warnings were
observed. Python compilation and `git diff --check` also passed.

The v3 checkpoint-continuation change adds exact-path, ordering, symlink,
YAML-injection, executable-wrapper, direct-qualification, AutoML-routing, and
SLURM-contract tests. Its focused campaign suite passed:

```text
46 passed
```

The complete v3 cross-model experiment suite passed:

```text
435 passed
```

The complete production and experiment repository suite passed:

```text
970 passed, 1 skipped
```

Only the three established sklearn Gaussian-process convergence warnings were
observed in each relevant complete suite.
