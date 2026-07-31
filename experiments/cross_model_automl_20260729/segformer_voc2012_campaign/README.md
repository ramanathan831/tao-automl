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

After the dataset is read-only, direct-full-run PTM qualification is complete,
and supported registry changes have been independently reviewed:

```bash
cd /localhome/local-rarunachalam/tao-automl
python -m experiments.cross_model_automl_20260729.segformer_voc2012_campaign.manifest_generator \
  --output /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/segformer_voc2012_three_mode/campaign.v1.json

python -m experiments.cross_model_automl_20260729.segformer_voc2012_campaign.run_campaign \
  --contract /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/segformer_voc2012_three_mode/campaign.v1.json \
  --automatic-trigger \
  --launch
```

No confirmation is required after prerequisites pass. The first real
candidate in each mode runs in parallel; the remaining 29 candidates per mode
are released automatically only when all three first candidates pass training,
standalone validation, stabilized latency, audit, and provenance gates.
