# SegFormer / full VOC2012 campaign

This directory freezes a direct production campaign for TAO network identifier
`segformer` and the complete official VOC2012 semantic-segmentation train/val
splits. It does not contain a CPU model run, model smoke test, mini-step, or
synthetic-input benchmark.

The campaign launches three independent Bayesian AutoML controllers:

| Mode | Recommendation target | Final policy |
| --- | --- | --- |
| Accuracy | Expected improvement on `val_miou` | Highest valid accuracy |
| Latency | Constrained expected improvement | Raw-minimum-anchored equivalent-fastest cohort at 90% retained accuracy |
| Multi-objective | ParEGO expected improvement on mIoU and latency | Independent rank-zero augmented-Chebyshev compromise |

Every train, standalone validation, and latency child requests one node and
eight A100 GPUs and runs directly from the pinned TAO 7.1 SQSH. Latency uses
real validation images, 50 warm-ups, five rounds of 100 model-forward samples
on each of eight replicas, and a 4,000-sample quality gate.

A finite `val_miou` below 0.10 halts the first-candidate gate for data,
optimization, fidelity, and metric root-cause analysis. This preregistered
VOC experiment sanity gate is not an AutoML selection constraint.

## Fail-closed PTM state

The repository currently records 13 official SegFormer PTMs, all as
`unverified`. The campaign does not reinterpret that state. Each arm must first
complete a real full-dataset, ten-epoch, one-node/eight-GPU train and standalone
validation workflow. A successful arm becomes eligible only after its exact
repository registry record is independently promoted to `supported`; terminal
failures remain exclusions. The runtime never mutates or bypasses the registry.

Consequently, the automatic trigger waits rather than launching an unsupported
PTM. This is the current intentional blocker, not an agent-selected PTM.

`qualification_campaign.py` implements that missing qualification step. Its
data-only stage resolves all 13 exact NGC members, verifies their immutable
identities and registered sizes, generates the checkpoint-target-specific train
and evaluation YAMLs from the packaged SegFormer templates, checksums them, and
publishes every checkpoint and spec read-only on Lustre. It does not import a
model framework, load a checkpoint, or construct a scheduler job.

The launch phase re-hashes the complete stage and submits exactly 13 independent
workflows. Each workflow runs full VOC2012 training for ten epochs with
validation every epoch, resolves its exact terminal checkpoint, and then runs
standalone evaluation over the complete validation split. Each train and
evaluation job uses one node, eight `NVIDIA A100-SXM4-80GB` GPUs, and the pinned
SQSH. `polar3` is capped at four hours, so the controller freezes the
skill-compliant `4.0`-hour scheduler limit and `3.8`-hour SDK timeout.

All arms are attempted. A failed arm is retained as a terminal structured
failure; it is never replaced with a fallback checkpoint. Completion
automatically writes both the exact `qualification_gate.py` input and an
independent-registry-review handoff. The controller does not promote registry
records itself. The gate binds the pre-promotion stage to immutable PTM source,
size, architecture, backbone, task, target field, and observed checkpoint
checksum, while reading eligibility from the independently promoted current
registry. Thus a status/validation promotion cannot invalidate genuine
qualification evidence or silently change the qualified checkpoint bytes.

## Frozen data

The prepared root is:

```text
/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/cross_model_automl_20260729/voc2012_segmentation_v1/prepared
```

It contains 1,464 train and 1,449 validation image/mask pairs. Masks retain
VOC IDs 0–20 and ignore ID 255. TAO uses `label_transform: "None"`,
`num_classes: 21`, and an explicit grayscale palette. The byte-identical local
and Lustre stage record is frozen at
`437ff12490637950707b9b951d820ea34d38b926080a478a5d182c2d284a0c5d`;
all 5,827 manifest entries passed and the remote dataset has zero writable
files.

## Reproduction

First seal the clean integrated source into an external campaign contract:

```bash
cd /localhome/local-rarunachalam/tao-automl
PYTHONPATH="$PWD/src" \
python -m experiments.cross_model_automl_20260729.segformer_voc2012_campaign.manifest_generator \
  --output /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/segformer_voc2012_three_mode/campaign.v1.json
```

Stage and independently verify all PTM/spec inputs without reserving GPUs:

```bash
PYTHONPATH="$PWD/src" \
python -m experiments.cross_model_automl_20260729.segformer_voc2012_campaign.qualification_campaign \
  --stage

PYTHONPATH="$PWD/src" \
python -m experiments.cross_model_automl_20260729.segformer_voc2012_campaign.qualification_campaign \
  --check-stage
```

The following explicit command is the only qualification path that submits
jobs. It starts all 13 independent direct-full-run workflows; no smoke or
mini-step precedes them:

```bash
PYTHONPATH="$PWD/src" \
python -m experiments.cross_model_automl_20260729.segformer_voc2012_campaign.qualification_campaign \
  --launch
```

After successful records have been independently reviewed and promoted in the
repository registry, rebuild the production wheel and reseal the campaign
contract against that clean commit. Then start the automatic three-mode
trigger:

```bash
PYTHONPATH="$PWD/src" \
python -m experiments.cross_model_automl_20260729.segformer_voc2012_campaign.run_campaign \
  --contract /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/segformer_voc2012_three_mode/campaign.v1.json \
  --automatic-trigger \
  --launch
```

No confirmation is required after prerequisites pass. The first real
candidate in each mode runs in parallel; the remaining 29 candidates per mode
are released automatically only when all three first candidates pass training,
standalone validation, stabilized latency, audit, and provenance gates.
