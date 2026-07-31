# Mask Grounding DINO / full COCO 2017 AutoML campaign

This directory prepares three independent, objective-aware AutoML jobs for the
exact TAO model identifier `mask_grounding_dino`. It does not run a CPU model
test, smoke test, mini-step, local GPU job, or SLURM job. Every eventual model
job is bound to the pinned TAO 7.1 SQSH and one node with eight A100 GPUs.

The campaign is statically prepared but is not launch-ready. All four official
Mask Grounding DINO checkpoints remain `unverified`; their immutable stage,
direct full-run qualification evidence, and reviewed registry promotion do not
exist yet. The automatic trigger fails closed until those prerequisites exist.

## Scientific contract

This is category-prompted COCO instance segmentation through Mask Grounding
DINO's `data_type: OD` path. The prompts are the 80 COCO category names. This
does not claim phrase-grounding coverage.

The primary metric is `segm_val_mAP50_95`, read from TAO's exact
`[segm] val_mAP@50-95` status key. The inspected TAO 7.1 implementation creates
an OD evaluator with both `bbox` and `segm` IoU types. VG `overall_IoU` is a
different referring-expression metric and is never accepted as COCO mask AP.
The repository metric policy records the mask metric as an unscaled fraction
in `[0, 1]`.

## Frozen data and text assets

The complete official COCO 2017 stage is read-only at:

```text
/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/cross_model_automl_20260729/coco2017_instance_panoptic_v1
```

Training uses the official TAO COCO-to-ODVG projection:

```text
tao/mask_grounding_dino/train/instances_train2017_odvg.jsonl
tao/mask_grounding_dino/train/instances_train2017_odvg_labelmap.json
```

The projection contains 117,266 annotated training images and preserves all
860,001 instance annotations and all 860,001 masks. The underlying full COCO
stage still contains all 118,287 train images. The ODVG JSONL SHA-256 is
`d5deb4f5cfe027786fb1ceb52632ad6d3ef027e95e434525ba715d6841fb2921`;
the label-map SHA-256 is
`02075d96f6bf06d061f9329b4775dc7c3bb5ac140c77bc5c0e465d305c46d6c1`.

TAO's OD evaluator requires contiguous category IDs. Validation and standalone
evaluation therefore use the deterministic, lossless annotation derivative:

```text
/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/cross_model_automl_20260729/mask_grounding_dino_coco2017_od_v1/instances_val2017_remapped.json
```

Its SHA-256 is
`9c9af9918e29292adfaa78a694d471e2be6d226e150300d9f4b22c2d77723ebc`.
It contains the same 5,000 images, 36,781 annotations, and mask segmentations as
the official validation JSON, with category IDs remapped to `0..79`. The
conversion manifest is
`coco2017_contiguous_validation.v1.json`, SHA-256
`3c2d09d20211017575a2c51a6797ef91f1939340d978a5d11d1edab1a30b2d`.
Both staged derivative files are read-only, and a repeated conversion was
byte-identical.

The original full-stage manifest SHA-256 is
`437ff12490637950707b9b951d820ea34d38b926080a478a5d182c2d284a0c5d`;
the 246,593-entry file-manifest SHA-256 is
`10566a60498de9998154f44a34445a488c9f030e09f2a7346d20a4a1c55f804e`.

The frozen offline BERT asset is:

```text
/lustre/fsw/portfolios/edgeai/users/rarunachalam/ptms/huggingface/bert-base-uncased/86b5e0934494bd15c9632b12f734a8a67f723594
```

Its five-file tree SHA-256 is
`04cd5cc67804f4752df93e7c05dd51d904e82fc05d28794ddb03504cca689fb5`.
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are mandatory.

## Official checkpoint inventory and qualification

The repository-owned inventory contains these four official Swin-T arms:

```text
mask_grounding_dino.commercial.swin_tiny.trainable.v2.1
mask_grounding_dino.commercial.swin_tiny.trainable.v2.0
mask_grounding_dino.commercial.swin_tiny.trainable.v1.0
mask_grounding_dino.research.swin_tiny.trainable.v2.0
```

Each record has an exact immutable NGC identity, expected member size,
TAO-7.1 compatibility declaration, license/access metadata, and a
repository-owned path-free YAML sidecar. Known checkpoint hashes are retained;
the older commercial v1.0 member must be hashed while staging. All four remain
`unverified` and are not runtime eligible.

Qualification is deliberately stronger than a smoke test. Each staged arm must
complete one real three-epoch full-dataset train and standalone full-validation
workflow on one node/eight A100s. In-epoch and standalone mask AP must be finite
and pass the preregistered experiment sanity gate of `0.05`. Failures are
terminal preserved exclusions. Successful evidence does not mutate or bypass
the registry: the exact record must then be independently reviewed and promoted
to `supported`.

## Frozen objective-aware search

PTM identity is a hierarchical, non-ordinal outer arm. Within each arm, the
search space is:

```yaml
model.num_select: [50, 100, 200, 300]
train.optim.lr: [1.0e-5, 5.0e-4]          # log scale
train.optim.lr_backbone: [1.0e-6, 5.0e-5] # log scale
train.optim.weight_decay: [1.0e-5, 1.0e-3] # log scale
```

`model.enc_layers` and `model.dec_layers` are fixed at six by the packaged TAO
schema and are not searched. Each mode gets 24 recommendations, three complete
training epochs per candidate, search seed `271828`, and training seed `1234`.
The modes have separate empty observation namespaces and share no observations.

| Mode | Recommendation acquisition | Terminal selection |
| --- | --- | --- |
| Accuracy | Expected improvement on mask AP | Highest valid mask AP |
| Latency | Constrained expected improvement using a monotonic best-observed in-job reference | Highest-accuracy member of the raw-minimum-anchored equivalent-fastest cohort satisfying 90% retained accuracy |
| Multi-objective | ParEGO expected improvement on mask AP and latency | Pareto-rank-zero, front-normalized augmented-Chebyshev compromise |

Latency retention applies only to latency mode. Multi-objective mode does not
inherit it. All agent-intervention and validation-to-selection feedback flags
are frozen `false`.

## Selection-time latency

Every candidate uses the same 16 immutable real COCO validation images, 80
frozen prompts, FP32, and batch size one. The timed scope includes Mask
Grounding DINO model forward and GPU mask postprocessing. Checkpoint loading,
disk I/O, image decode, resize/normalization, host-to-device transfer, text
tokenization, mask serialization, metric accumulation, and distributed gather
are excluded.

Each one-node allocation runs 50 warm-ups and five rounds of 100 requests on
eight synchronized replicas: 4,000 raw samples per candidate. Median, p95, MAD,
IQR, robust CV, bootstrap interval, round drift, and device-spread gates must
pass.

## Automatic launch gate

The final campaign is sealed only after source, wheel, SDK, skills, SQSH,
dataset, BERT, PTM stage, qualification evidence, and supported registry status
all match their frozen identities. `--automatic-trigger --launch` then starts
the three independent mode controllers without requesting confirmation.
Candidate zero in all three modes must pass full train, standalone evaluation,
latency, provenance, and audit gates before the remaining 23 recommendations
per mode are released automatically.

Current exact blockers:

- the four-PTM immutable stage manifest is absent;
- direct full-run qualification evidence is absent;
- all four registry records remain `unverified`;
- a clean post-change source commit and matching production wheel have not been
  sealed into `campaign.v1.json`.

No campaign manifest is committed because sealing it before those identities
exist would create invalid evidence.

## Reproduction sequence

After staging the exact four PTMs and building a wheel from a clean reviewed
commit:

```bash
cd /localhome/local-rarunachalam/tao-automl
export PATH=/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin:$PATH

python -m \
  experiments.cross_model_automl_20260729.mask_grounding_dino_coco2017_campaign.manifest_generator \
  --output \
  /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/mask_grounding_dino_coco2017_three_mode/campaign.v1.json

# Plan-only: constructs no scheduler client and submits no job.
python -m \
  experiments.cross_model_automl_20260729.mask_grounding_dino_coco2017_campaign.qualification_campaign \
  --contract \
  /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/mask_grounding_dino_coco2017_three_mode/campaign.v1.json

# Add --launch only for the reviewed direct-full qualification run.
# After successful evidence and independent registry promotion, reseal the
# final campaign against the promoted clean source and wheel.

python -m \
  experiments.cross_model_automl_20260729.mask_grounding_dino_coco2017_campaign.run_campaign \
  --contract \
  /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/mask_grounding_dino_coco2017_three_mode/campaign.v1.json \
  --automatic-trigger \
  --launch
```
