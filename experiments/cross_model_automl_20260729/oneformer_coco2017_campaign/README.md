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
overlay at commit `1752ec2c2a7040d4db0e6c3e6f52cc489e8dbc86`, archive SHA-256
`a3d71c97c3a5fe9c2cf3c44e778681d0b8d6eb16475e0b64c8f3c2819446a074`.
Every training, standalone-evaluation, and latency command verifies and
installs that overlay before importing TAO PyTorch, and persists an installer
receipt. A missing, changed, or inapplicable overlay leaves the automatic
trigger closed.

The first GPU qualification (`v1`) exposed a launcher defect: the overlay
prefix was appended directly to the SDK command, so its shell operators were
not contained inside the Pyxis container.  The installer therefore inspected
the allocation host instead of the pinned SQSH and the unpatched OneFormer
schema rejected `evaluate.task`.  All four v1 failures are preserved.  The v2
launcher executes the complete overlay-plus-entrypoint payload through one
quoted in-container `bash -lc`; no model, dataset, PTM, metric, budget, or gate
setting changed.

All four independent v2 qualification workflows were submitted and preserved
as terminal failures. Their sealed completion record is
`/localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/oneformer_coco2017_ptm_qualification_v2/completion.json`
(SHA-256
`9e7e059ae7bea812b391c2eff82bab4aa888aa7511460ae5676a02a6f8059cd1`).
The common failure occurred before training: the v2 installer audited the
empty ephemeral output tree instead of the immutable package root in the
pinned SQSH. In addition, the SDK entrypoint begins with a best-effort install
ending in `|| true`; without grouping the complete entrypoint, that clause
could mask an overlay-prefix failure and allow unpatched TAO to start.

The versioned v3 qualification fixes only those launcher defects. Overlay
manifest/receipt schema 2 audits
`/usr/local/lib/python3.12/dist-packages` directly, writes patched modules to a
separate ephemeral `PYTHONPATH` tree, and groups the complete SDK entrypoint on
the right side of the fail-closed overlay `&&`. The model, dataset, four PTM
arms, metric, one-epoch budget, and eight-A100 resource contract remain
unchanged. No v3 replacement is submitted by preparation or sealing; the
automatic campaign release remains closed until exact direct-full
qualification evidence is successful and accepted by the qualification gate.

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
  --runtime-overlay /localhome/local-rarunachalam/.tao/artifacts/oneformer-runtime-product-fixes-1752ec2c/oneformer-runtime-overlay.v2.tar \
  --output /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/oneformer_coco2017_three_mode_v3/campaign.v3.json

python -m experiments.cross_model_automl_20260729.oneformer_coco2017_campaign.run_campaign \
  --contract /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/oneformer_coco2017_three_mode_v3/campaign.v3.json \
  --runtime-root /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/oneformer_coco2017_three_mode_v3 \
  --automatic-trigger \
  --launch
```

No post-gate confirmation is required. The automatic trigger itself performs
the transition once immutable prerequisites pass. This preparation path does
not run a CPU model, model smoke, mini-step, GPU model, or SLURM job.
