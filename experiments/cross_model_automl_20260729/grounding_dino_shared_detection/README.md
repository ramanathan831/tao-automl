# Grounding DINO shared synthetic detection preparation

This directory prepares, but does not launch, the exact TAO model identifier
`grounding_dino` on the same immutable synthetic COCO detection dataset used
for DINO, Deformable DETR, and RT-DETR.

The source is a valid category-detection corpus. NVIDIA TAO Data Services can
derive ODVG detection records and a label map from the four COCO category
names, and can remap validation IDs from `1..4` to the contiguous `0..3`
contract required by Grounding DINO. The prompt list is derived verbatim and
in category-ID order:

`cone`, `forklift`, `cart`, `fire_extinguisher`

No synonym, phrase, candidate, PTM, or preferred label is injected by an
agent. The campaign is explicitly category-prompted open-vocabulary detection
with `val_mAP50`; it is not a referring-expression grounding campaign.

The source contains zero image `caption` fields and zero annotation
`tokens_positive` fields. It therefore cannot support a phrase-grounding or
`Pr@0.5` product claim. Production also has no supported
`grounding_dino`/`val_mAP50` metric-sanity policy, and both official repository
PTMs remain `unverified`. The automatic gate consequently remains closed.

Prepared execution is one direct full 10-epoch, one-node/eight-GPU
qualification per official PTM, followed only after evidence-backed PTM
promotion by three independent objective-aware AutoML jobs:

- accuracy: expected improvement on validation accuracy;
- latency: constrained expected improvement with a monotonically
  self-calibrated 90% retained-accuracy reference;
- multi-objective: ParEGO expected improvement over accuracy and latency.

Every model job uses the pinned TAO 7.1.0 RC245 `.sqsh` directly. There are no
CPU model runs, smoke runs, mini-steps, shared archives, manually injected
candidates, or scheduler submissions in this preparation.

Generate and verify the immutable preparation record with:

```bash
PYTHONPATH=src \
  python -m experiments.cross_model_automl_20260729.grounding_dino_shared_detection.prepare_campaign
```

The generated record intentionally states `launch_authorized: false`. A later
automatic successor may open the gate only after the converted dataset is
sealed, a category-detection metric policy is supported, and at least one
official PTM is qualified by full train/validation/evaluation evidence.
