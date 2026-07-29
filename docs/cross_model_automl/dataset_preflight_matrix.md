# Cross-model AutoML dataset and execution preregistration

Status: proposed Phase-4 preregistration, frozen before dataset acquisition,
model preflight, or GPU execution.

Audit date: 2026-07-29

This document is a planning and correctness gate. It is not evidence that any
dataset is presently staged, any model has passed preflight, or any campaign is
ready to launch.

During this audit:

- no dataset was downloaded;
- no checkpoint was downloaded;
- no model training, evaluation, inference, or latency benchmark ran;
- no local or SLURM GPU job was submitted;
- no experiment artifact or winner was changed;
- no dataset choice was made after observing cross-model results.

The dataset, split, task metric, and conversion contracts below must be frozen
and hashed before a model can pass local preflight. A model may proceed to
SLURM only after its complete local gate is green.

## Exact TAO model contracts

The implemented model identifiers, not conversational aliases, are:

| Model | Exact identifier | Task intended by this preregistration | Packaged data format |
| --- | --- | --- | --- |
| DINO | `dino` | Object detection | COCO |
| Deformable DETR | `deformable_detr` | Object detection | COCO |
| RT-DETR | `rtdetr` | Object detection | COCO |
| Grounding DINO | `grounding_dino` | Referring-expression box grounding | ODVG for training; current validation path is COCO-only |
| SegFormer | `segformer` | Semantic segmentation | TAO/UNet-style image and mask folders |
| OneFormer | `oneformer` | Panoptic segmentation | COCO panoptic |
| Mask2Former | `mask2former` | Instance segmentation | COCO instance plus panoptic assets |
| Mask Grounding DINO | `mask_grounding_dino` | Referring-expression segmentation | ODVG/VG with masks |

All eight current model-skill records resolve the nominal default image to
`nvcr.io/nvidia/tao/tao-toolkit:7.0.1-pyt`. That tag must not be assumed to
contain changes from a local release/7.1.0 branch. Preflight must pin and hash
an SQSH whose installed wheel or mounted source identity matches the campaign
source commit.

## Proposed authoritative dataset matrix

The estimated epoch times below are capacity-planning ranges, not
measurements. They assume one A100, a model-supported batch size and crop, and
one complete pass over the stated training split. The first successful local
epoch must replace them with measured timing before campaign resource
manifests are frozen.

| Model | Task | Proposed complete dataset | Dataset size and official splits | Annotation contract | Primary metric | License and access | Estimated one-A100 epoch | Why selected |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- |
| `dino` | Object detection | PASCAL VOC2007 | 9,963 images, 24,640 objects, 20 classes; train 2,501, validation 2,510, test 4,952; approximately 450 MB train/validation plus 430 MB annotated test | VOC XML converted losslessly to canonical COCO bbox JSON | `val_mAP50` (COCO AP50), with `val_mAP` secondary | Public challenge data; constituent Flickr/MSRC image terms apply rather than one blanket open-source license | 15–35 min | Complete, moderate, multi-object scenes; public test annotations |
| `deformable_detr` | Object detection | PASCAL VOC2007 | Same complete corpus and splits | Same canonical COCO archive | COCO bbox AP@[0.50:0.95], with AP50 secondary | Same VOC image-specific terms | 10–30 min | Enables a fair shared detection corpus without forcing a subset |
| `rtdetr` | Object detection | PASCAL VOC2007 | Same complete corpus and splits | Same canonical COCO archive | COCO bbox AP@[0.50:0.95], with AP50 secondary | Same VOC image-specific terms | 5–20 min | Moderate enough for repeated AutoML while retaining realistic multi-object scenes |
| `grounding_dino` | Referring-expression box grounding | RefCOCOg, UMD split | 25,799 images, 49,822 referred objects, 95,010 expressions; train 21,899 images/42,226 objects/80,512 expressions, validation 1,300/2,573/4,896, test 2,600/5,023/9,602; image-disjoint UMD splits | RefCOCOg expression and COCO annotation identity converted to VG-style ODVG while preserving expression, bbox, source IDs, and image identity | Percentage of expressions whose predicted box has IoU >= 0.5 (`Pr@0.5`); mean box IoU secondary | RefCOCOg annotations originate from Google RefExp, whose official author release states CC BY 4.0; `refer` API code is Apache-2.0; underlying COCO images retain their source-image licenses | 2–6 h | A complete, task-correct grounding corpus rather than category-prompted detection |
| `segformer` | Semantic segmentation | Cityscapes fine annotations | 5,000 fine 2048x1024 frames; train 2,975, validation 500, test 1,525; 30 annotated classes and 19 evaluation train IDs; approximately 11 GB images plus fine labels | Official train-ID PNGs packaged as `images/<split>` and `masks/<split>`; ignore ID 255 preserved | mIoU over the 19 evaluation classes | Registration required; custom Cityscapes terms permit scientific, non-commercial use and prohibit redistribution | 20–60 min | Complete high-resolution semantic segmentation benchmark of manageable scale |
| `oneformer` | Panoptic segmentation | Cityscapes fine annotations | Same complete fine corpus and official splits | Official Cityscapes-to-COCO panoptic conversion, panoptic PNGs, JSON, and label map | Panoptic Quality (PQ); PQ-things and PQ-stuff secondary | Same registered, non-commercial Cityscapes terms | 45–120 min | Task-correct panoptic labels are available from the same authoritative source |
| `mask2former` | Instance segmentation | Cityscapes fine annotations | Same complete fine corpus; official thing-instance annotations | COCO instance JSON plus the panoptic assets required by the TAO schema | COCO mask AP@[0.50:0.95] | Same registered, non-commercial Cityscapes terms | 45–120 min | Provides complete instance masks without introducing a different visual domain |
| `mask_grounding_dino` | Referring-expression segmentation | RefCOCOg, UMD split, joined to COCO instance masks | Same complete 25,799-image, 95,010-expression corpus and image-disjoint UMD splits | Each expression retains its RefCOCOg IDs, COCO object ID, bbox, and original polygon/RLE mask in VG-style ODVG | Overall/cumulative IoU (`overall_IoU`); mean IoU and Pr@0.5 secondary | RefCOCOg annotation CC BY 4.0 provenance plus source-image COCO licensing | 4–10 h | Task-correct language-conditioned segmentation with complete source masks |

Every detection metric uses the TAO COCO evaluator after a lossless
VOC-to-COCO conversion; none is the historical VOC2007 11-point AP. DINO uses
the release/7.1 skill's explicit AutoML recommendation, `val_mAP50`, so the
new pilot remains comparable with the completed DINO evidence and the
implemented metric extractor. `val_mAP` remains a required secondary metric.
The Deformable DETR and RT-DETR campaigns retain preregistered COCO
AP@[0.50:0.95] as their primary metric unless their local product contract
fails that extraction before any campaign is launched.

## Authoritative source and license records

### PASCAL VOC2007

- Overview, statistics, and database rights:
  <https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/>
- Project usage guidance:
  <https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/>
- Official train/validation archive:
  <https://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar>
- Official annotated test archive:
  <https://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar>

The official page reports 9,963 images and 24,640 annotated objects. It lists
the train/validation archive as approximately 450 MB and annotated test archive
as approximately 430 MB. The dataset contains images from multiple sources,
including Flickr and MSRC. The dataset card must preserve those source-specific
rights and must not describe VOC as a single permissively licensed corpus.

### RefCOCOg and COCO 2014

- Official Google RefExp author toolbox and license statement:
  <https://github.com/mjhucla/Google_Refexp_toolbox>
- Original Google RefExp paper:
  <https://arxiv.org/abs/1511.02283>
- Official Google release identity:
  <https://storage.googleapis.com/refexp/google_refexp_dataset_release.zip>
- Referring-expression API, which recommends the image-disjoint UMD split:
  <https://github.com/lichengunc/refer>
- COCO source:
  <https://cocodataset.org/>

The Google author toolbox states that its RefExp data is CC BY 4.0 and that
COCO 2014 images and annotations are required. It describes the COCO image
download as approximately 13 GB and the annotations as approximately 158 MB.
COCO images retain their individual source-image licensing.

The UMD split is preregistered because it supplies image-disjoint train,
validation, and test populations. The current `refer` README also warns that
its historical download server is broken. The Google storage URL returned an
access error during this read-only audit. Before adoption, dataset preflight
must resolve an authorized immutable source, prove that it is the intended
release, and record its SHA256. An unpinned community mirror is not an
acceptable silent replacement.

### Cityscapes

- Official overview:
  <https://www.cityscapes-dataset.com/dataset-overview/>
- Authenticated download portal:
  <https://www.cityscapes-dataset.com/downloads/>
- Dataset terms:
  <https://www.cityscapes-dataset.com/license/>
- Official conversion and evaluation scripts:
  <https://github.com/mcordts/cityscapesScripts>

Required upstream packages are `leftImg8bit_trainvaltest.zip` and
`gtFine_trainvaltest.zip`. The official scripts report 2,975 training, 500
validation, and 1,525 test images. Test labels are not public. The repository
provides the authoritative semantic train-ID, instance-ID, panoptic conversion,
and semantic, instance, and panoptic evaluation tools.

Cityscapes requires registration and permits scientific non-commercial use.
It prohibits redistribution of the dataset and reconstructable derivatives.
Dataset staging must remain within the licensed user and cluster boundary.

## Split and final-evaluation policy

- VOC2007 keeps the official train, validation, and annotated test lists
  unchanged. AutoML observes validation metrics; the test split remains
  untouched until final frozen-candidate evaluation.
- RefCOCOg uses the complete, image-disjoint UMD train, validation, and test
  lists unchanged.
- Cityscapes keeps the official train and validation sets unchanged for the
  initial correctness preflight. Because public test labels are unavailable,
  a cross-model campaign that requires an accuracy holdout independent of
  AutoML validation must preregister an additional group-disjoint split from
  official training data and reserve the official validation set for terminal
  evaluation. The exact list, seed, grouping rule, and hash must exist before
  any recommendation. It may not be created after pilot results.

No test or terminal-evaluation metric may feed recommendation, feasibility,
selection, or reselection.

## Existing TAO Data Services capability

The local `tao-dataservices` source at audit time was:

```text
dcea3a39bd3e4709e2325e4b61a4f179efebde4c
```

Reusable implementation exists under
`nvidia_tao_ds/annotations/conversion/`:

- COCO to ODVG;
- ODVG to COCO;
- COCO category remapping to contiguous IDs;
- COCO to KITTI;
- KITTI to COCO.

`coco_to_odvg.py`:

- emits category-detection or caption/token-span grounding records;
- converts COCO `xywh` boxes to ODVG `xyxy`;
- preserves polygon or RLE `segmentation` values as ODVG masks;
- writes a label map for detection data;
- remaps present category IDs to a contiguous ODVG label domain.

The COCO and KITTI data-analytics paths can validate common box failures and
produce statistics. They should be reused rather than duplicated in the
AutoML repository.

### Existing capability that is not yet sufficient

The current COCO/ODVG tests principally prove that output files and
annotations exist. They do not prove:

- source-to-output image and annotation count equality;
- category-map bijection;
- bbox round-trip equality;
- decoded polygon/RLE mask equality;
- source file and dimension identity;
- expression, token-span, and source-ID preservation;
- deterministic output independent of input enumeration.

The `use_all_categories=true` branch in the current COCO-to-ODVG converter
indexes a category-count array using raw category IDs. That is unsafe for
non-contiguous category IDs. Until fixed and tested, the preregistered path
must use the default `use_all_categories=false` and separately audit categories
with zero instances.

Optional correction in the COCO validator is inappropriate for immutable
campaign preparation. Validation must fail closed and report an error rather
than silently modifying source-derived annotations.

## Required conversion and validation work

The following repository-owned, tested preparation paths are still absent:

1. VOC XML to canonical COCO detection JSON.
2. Native RefCOCOg plus COCO annotation IDs to expression-preserving VG ODVG.
3. Cityscapes train-ID masks to the exact SegFormer folder, label-map, and
   palette contract.
4. Cityscapes instance IDs or polygons to COCO instance JSON required by
   Mask2Former.
5. Dataset acquisition manifests and dataset cards with authoritative source
   and archive hashes.
6. ODVG/VG integrity validation.
7. Semantic-mask, COCO-instance, and COCO-panoptic integrity validation.

The official Cityscapes scripts must be pinned and wrapped for semantic,
instance, and panoptic preparation instead of reimplementing their label
semantics.

### Conversion-specific requirements

#### VOC2007

- Preserve the official split lists.
- Preserve every image and non-difficult annotation according to one declared
  difficult-object policy.
- Use stable image, annotation, and category IDs.
- Preserve the canonical category names and their order.
- Report source and output object counts by split and class.
- Configure each model for the actual category IDs. DINO's class capacity must
  exceed the maximum retained ID; Deformable DETR must receive every
  `eval_class_id`; RT-DETR remapping must be explicit and audited.

#### RefCOCOg

- Preserve `ref_id`, `sent_id`, COCO `ann_id`, source image ID, expression,
  bbox, and segmentation.
- Prove UMD train, validation, and test image sets are disjoint.
- Emit one unambiguous evaluation record per expression and target.
- Do not reconstruct masks from boxes. Decode the original COCO polygon/RLE.
- Record empty or invalid source records as structured failures; do not drop
  them silently.

#### Cityscapes

- Preserve all 19 evaluation train IDs and ignore ID 255.
- Preserve the exact source-to-train-ID and thing/stuff mapping.
- Verify image-mask dimensions and pixel label domain.
- Use the official panoptic converter and evaluator.
- Derive instance JSON deterministically and prove mask area, bbox, category,
  and crowd semantics against the source instance-ID images.
- Keep the SegFormer palette and `label_transform` consistent across train,
  evaluate, inference, export, and reload.

## Per-model metric and readiness blockers

### DINO

Dataset design is ready for implementation, but no public dataset or PTM has
been staged or smoke-tested. The VOC category-ID contract and COCO mAP parsing
must pass before local epoch execution.

### Deformable DETR

The complete `eval_class_ids` set must be supplied. A finite aggregate metric
with silently omitted classes is not a passing preflight.

### RT-DETR

Category remapping and foreground/background indexing must be proved against
the canonical VOC COCO file. A numerically finite result with shifted labels is
invalid.

### Grounding DINO

Current source constructs `CocoDetection` for validation in
`nvidia_tao_pytorch/cv/grounding_dino/dataloader/pl_odvg_data_module.py`.
Training accepts ODVG, including phrase records, but validation reports COCO
bbox mAP from category captions.

Consequently, the current path can validate category-prompted open-vocabulary
detection but cannot establish the requested referring-expression grounding
claim. Grounding DINO is not SLURM-ready until a phrase-grounding validation
loader and Pr@0.5 evaluator are implemented and locally verified. Substituting
VOC category prompts must be labeled as detection and is not an acceptable
grounding result.

### SegFormer

Preflight must verify:

- 19 classes rather than the binary template default;
- ignore ID 255;
- exact grayscale mask IDs;
- palette and label transformation;
- task-correct mIoU after checkpoint reload.

### OneFormer

Panoptic inference exists, but inspected validation code reports semantic
mIoU and pixel accuracy. No training-time PQ path was found. A panoptic
campaign cannot optimize semantic mIoU and then claim panoptic quality.
OneFormer remains blocked until PQ is emitted, parsed, and independently
checked against the official Cityscapes evaluator.

### Mask2Former

The inspected validation path also reports semantic mIoU and pixel accuracy,
although instance and panoptic inference modes exist. No mask AP validation
path was found. Mask2Former remains blocked until COCO mask AP is emitted,
parsed, and checked against the official instance evaluator.

### Mask Grounding DINO

Current source code accepts `data_type: VG` for validation and its VG evaluator
returns:

- `mIoU`;
- `overall_IoU`;
- `mAP50`;
- `mAP`;
- target/no-target accuracy;
- Pr@IoU thresholds.

The packaged skill, however, instructs train-stage AutoML to use `val_loss` and
describes validation as COCO-format. The exact pinned container may therefore
not match the inspected source. The discrepancy is a hard preflight blocker.

The model-specific smoke contract must also keep:

- `model.enc_layers = 6`;
- `model.dec_layers = 6`;
- a non-degenerate query count, at least 100 for the generic smoke case.

The existing generic AutoML smoke ranges that reduce depths to one or two are
invalid for this model.

Mask Grounding DINO's VG evaluator reports percentage metrics on a 0-to-100
scale, while several TAO detection metrics are reported on a 0-to-1 scale.
The objective registry and sanity policy must declare metric scale and
direction explicitly.

## Dataset integrity gate

Every prepared dataset must produce a machine-readable report that proves:

1. Every downloaded archive matches its frozen SHA256.
2. Every extracted file belongs to a canonical manifest with a root hash.
3. Every referenced image exists, decodes, and matches declared dimensions.
4. Train, validation, test, and final-evaluation image identities obey the
   preregistered overlap policy.
5. Duplicate image, annotation, expression, and candidate IDs are rejected.
6. Boxes are finite, positive-area, in bounds, and in the declared coordinate
   convention.
7. Categories have a bijective, documented mapping and no class silently loses
   annotations.
8. Polygon and RLE masks decode successfully and match source area and pixel
   identity.
9. Semantic masks contain only declared labels and ignore IDs.
10. Panoptic segment IDs, JSON records, masks, areas, and thing/stuff flags
    agree.
11. Every grounding expression is nonempty and maps to the exact source object
    and mask.
12. Conversion output is deterministic and invariant to source enumeration
    order.
13. No source record is silently corrected, excluded, or relabeled.
14. Class, annotation, mask-area, image-size, and empty-sample statistics are
    recorded by split.
15. Dataset card, conversion config, converter commit, output manifest, and
    integrity report are all hash-linked.

When an upstream source publishes no checksum, preflight must calculate SHA256
over the untouched archive immediately after download and freeze it before
extraction. A mutable URL is not an immutable identity.

## Required local model preflight

The existing
`tao-automl/scripts/validate_skill_automl_model.py` is an internal small-dataset
smoke harness. It runs minimal AutoML recommendations and post-check
evaluation/inference, but it does not satisfy this campaign's local gate.

For each model, local single-GPU preflight must prove:

1. Complete prepared dataset passes the integrity gate.
2. Default registered PTM checksum and load succeed.
3. Every additional registered PTM loads and completes train, validation, and
   inference mini-steps.
4. Merged PTM YAML, AutoML profile, user values, and candidate overrides have
   the declared precedence.
5. One training batch succeeds.
6. One validation batch succeeds with a task-correct finite metric.
7. One inference batch succeeds.
8. One complete epoch over the full training split succeeds.
9. Validation runs within that epoch.
10. Standalone evaluation succeeds.
11. A checkpoint is saved, reloaded exactly, and produces a valid evaluation.
12. Resume behavior is demonstrated from a non-`latest` checkpoint.
13. Stabilized latency instrumentation succeeds under a frozen input and
    runtime contract.
14. Outputs, logs, configuration, PTM, dataset, image, and package identities
    are hash-linked.

No model is SLURM-ready merely because its configuration parses or a single
mini-batch is finite.

## Local readiness matrix at preregistration

`Pending` means the design is identified but no execution has occurred.
`Blocked` means a concrete implementation or access defect must be resolved
before execution.

| Model | Dataset contract | Task metric path | Conversion path | Access | Full local epoch | SLURM ready |
| --- | --- | --- | --- | --- | --- | --- |
| `dino` | Proposed | Pending | Missing VOC-to-COCO | Pending | Not run | No |
| `deformable_detr` | Proposed | Pending | Missing VOC-to-COCO | Pending | Not run | No |
| `rtdetr` | Proposed | Pending | Missing VOC-to-COCO | Pending | Not run | No |
| `grounding_dino` | Proposed | Blocked: no phrase-grounding validation metric | Missing RefCOCOg-to-VG ODVG | Blocked pending immutable RefCOCOg access | Not run | No |
| `segformer` | Proposed | Pending | Missing Cityscapes-to-TAO packaging | Registration pending | Not run | No |
| `oneformer` | Proposed | Blocked: no verified PQ | Official panoptic conversion reusable; TAO packaging pending | Registration pending | Not run | No |
| `mask2former` | Proposed | Blocked: no verified mask AP | Missing COCO-instance packaging | Registration pending | Not run | No |
| `mask_grounding_dino` | Proposed | Blocked: skill/container/source metric mismatch | Missing RefCOCOg-to-VG ODVG | Blocked pending immutable RefCOCOg access | Not run | No |

## Later SLURM and strict SQSH contract

This section records the launch contract for use only after local preflight and
the DINO three-job pilot pass. It does not authorize or describe a job launched
during this audit.

Follow the packaged `tao-run-on-slurm` workflow and use `SlurmSDK.create_job`
and `build_entrypoint`. Do not hand-roll campaign `sbatch` scripts.

### Data and storage

- Prestage complete datasets, PTMs, specs, and wheels on Lustre.
- Verify every required path with `test -e` from the login host.
- Use `lustre:///absolute/path` or its resolved absolute path in job inputs.
- Do not download S3, HTTP, Hugging Face, or NGC training data after a GPU
  allocation starts.
- Mount `/lustre` read-only for inputs and use an explicit writable results
  path.

### One-node, eight-GPU shape

Every full candidate job must request and use the entire node:

```text
num_nodes = 1
gpu_count = 8
train.num_gpus = 8
train.gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]
cpus_per_task = 16
```

Use the packaged four-hour/3.8-hour timeout defaults unless the selected
partition explicitly supports a longer frozen request. The SDK emits one
Pyxis task with `#SBATCH --gres=gpu:8`; the TAO/Lightning entrypoint creates
the local distributed workers.

### SQSH must fail closed

Set and verify:

```text
SLURM_USE_SQSH = true
```

Preconvert the pinned container on the CPU conversion partition, record the
absolute `.sqsh` path and SHA256, and use:

```text
srun --container-image=/absolute/pinned-image.sqsh --container-mounts=/lustre
```

At audit time,
`tao_sdk/platforms/slurm/handler.py::_prepare_container_image` caught SQSH
conversion failures and returned the registry image. That silent fallback
violates the campaign requirement. Before submission, production must either:

1. provide a strict-SQSH option that raises on conversion failure; or
2. accept only a pre-staged `.sqsh` and assert that the resolved job image is
   the exact hashed file.

A job that falls back to a registry pull must fail preflight and must not enter
the campaign.

### Retries and staged release

The SDK's audited constant was `MAX_JOB_RETRIES = 10`. The campaign must
preregister a smaller finite infrastructure retry budget and preserve every
failed training recommendation as a failed record. A failed candidate may not
be silently replaced to reach a requested successful count.

Submission remains staged:

1. one candidate per mode;
2. a small pilot batch after artifact validation;
3. the frozen full budget only after the pilot passes;
4. matched validation only after winners and relevant Pareto candidates are
   frozen.

## Preregistration conclusion

The dataset choices are scientifically appropriate and moderate relative to
the supported tasks, but this matrix does not mark any model ready for SLURM.
The phrase-grounding evaluator, panoptic PQ, instance mask AP, Mask Grounding
DINO metric contract, missing conversion paths, immutable dataset access, and
strict SQSH behavior are concrete gates.

No download, local model run, latency benchmark, or SLURM submission occurred
while producing this preregistration.
