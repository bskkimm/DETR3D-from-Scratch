# Full DETR3D Architecture Notebook Design

## Goal

Replace `notebooks/detr3d_e2e_walkthrough.ipynb` with a literate, executable
reimplementation of the faithful DETR3D neural architecture currently on
`main`. The notebook must teach and define the model rather than importing it
from `detr3d`, `train.py`, or `eval.py`.

The notebook is architecture-focused. Dataset loading, matching and loss,
training loops, checkpoint utilities, diagnostics, and official nuScenes
evaluation may be imported from the production package when demonstrating how
to train or evaluate the notebook-defined model.

## Success Criteria

The completed notebook must:

1. Define every component on the faithful DETR3D model forward path locally.
2. Preserve production class boundaries, parameter names, tensor ordering,
   initialization, and operation order closely enough for strict checkpoint
   loading.
3. Put each public submodule or major architectural chunk in its own code cell.
4. Put explanatory Markdown immediately before each implementation cell.
5. Annotate every dimension-changing code line with symbolic input/output
   shapes and faithful C6 dimensions where useful.
6. Include equations matching the code's variable names and index ordering.
7. Include an ASCII diagram for the full architecture and focused diagrams for
   projection/sampling, cross-attention, decoder refinement, and box encoding.
8. Execute from top to bottom in a reduced synthetic smoke mode without a
   dataset or checkpoint.
9. Optionally run one real nuScenes sample and strict faithful-checkpoint
   inference when local assets are configured.
10. Contain no stored outputs or execution counts when committed.

## Scope

### Implement Locally

The notebook will define these components without importing project code:

- `GridMask`
- `ModulatedDeformConv2dPack`
- `MultiViewImageBackbone`
- `ImageFPN`
- `inverse_sigmoid`
- `denormalize_reference_points`
- projection-matrix and image-shape stacking helpers
- `feature_sampling`
- `Detr3DCrossAttention`
- `Detr3DDecoderLayer`
- `Detr3DTransformer`
- `MLP`
- `Detr3DHead`
- `Detr3DModel`
- `encode_bbox_targets`
- `decode_bbox_predictions`
- a local model builder matching the faithful C6 construction

The implementation authority is the corresponding source under
`detr3d/models/` at the feature branch's base commit. The notebook must mirror
that behavior, including:

- official Caffe-style ResNet stride placement
- modulated DCNv2 in the final two ResNet stages
- frozen stem and evaluation-mode BatchNorm behavior
- four-level FPN with ReLU before the extra P6 convolution
- learned camera and feature-level cross-attention weights
- dedicated learned initial reference points
- six decoder layers
- iterative center refinement in inverse-sigmoid space
- detached references between decoder layers
- classification-head normalization and initialization
- semantic 9D and encoded 10D box conventions

### Imports Allowed For Demonstrations

Training and evaluation sections may import:

- `NuScenesDetr3DDataset` and collation helpers
- `HungarianMatcher3D` and `Detr3DLoss`
- trainer and optimizer/scheduler helpers
- FCOS3D initialization and checkpoint helpers
- diagnostic visualization helpers
- official nuScenes decoding, export, and evaluation helpers

These imports must consume the notebook-defined model. They must not replace or
bypass any architecture component listed in the local implementation scope.

### Excluded From Local Reimplementation

The notebook does not need to duplicate:

- nuScenes JSON table parsing and annotation conversion
- Hungarian assignment and training-loss internals
- MLflow, ETA, thermal monitoring, and recovery internals
- command-line argument parsing
- nuScenes submission conversion and devkit implementation

These topics may be explained and demonstrated through production imports.

## Notebook Organization

Each architecture section follows this pattern:

```text
Markdown: purpose, contract, equations, shape table, ASCII diagram
Code:     complete local implementation
Code:     reduced deterministic shape/behavior probe when useful
```

Closely related private helpers may share their public component's code cell.
Public submodules and major runtime chunks must remain in separate cells.

### Proposed Cell Sequence

1. Title, reproduction result, and notebook contract.
2. Full architecture ASCII diagram.
3. Imports, deterministic seed, execution mode, symbols, and configuration.
4. Optional production data/loss/training/evaluation imports.
5. Faithful dimensions and end-to-end tensor contract.
6. GridMask explanation and equations.
7. `GridMask` implementation.
8. GridMask smoke probe.
9. Caffe ResNet and modulated DCNv2 explanation and stage diagram.
10. `ModulatedDeformConv2dPack` implementation.
11. `MultiViewImageBackbone` implementation.
12. Backbone structure and reduced-shape probe.
13. FPN equations and top-down diagram.
14. `ImageFPN` implementation.
15. FPN shape probe.
16. Reference-point normalization equations.
17. `inverse_sigmoid` and `denormalize_reference_points` implementation.
18. LiDAR-to-image projection derivation and ASCII diagram.
19. Projection metadata helpers and `feature_sampling` implementation.
20. Deterministic feature-sampling geometry probe.
21. Learned camera/level cross-attention equations and diagram.
22. `Detr3DCrossAttention` implementation.
23. Cross-attention shape and mask probe.
24. Decoder operation-order equations and diagram.
25. `Detr3DDecoderLayer` implementation.
26. Decoder-layer smoke probe.
27. Query initialization and transformer stack explanation.
28. `Detr3DTransformer` implementation.
29. Regression field ordering and reference-refinement equations.
30. `MLP` and `Detr3DHead` implementation.
31. Head initialization, output, and refinement probe.
32. Iterative six-layer model ASCII diagram.
33. `Detr3DModel` implementation.
34. Local faithful model builder.
35. Reduced end-to-end architecture smoke forward.
36. Semantic 9D versus encoded 10D explanation and diagram.
37. Local box encode/decode implementation.
38. Box round-trip and yaw probe.
39. Production dataset and batch integration demonstration.
40. Production matching/loss integration demonstration.
41. One reduced forward/loss/backward/optimizer-step demonstration.
42. Production FCOS3D initialization and checkpoint-loading explanation.
43. Optional strict faithful-checkpoint load and one-sample inference.
44. Diagnostic visualization demonstration.
45. Official decoding/evaluation explanation and optional execution.
46. Architecture parity report and verification commands.
47. Faithful training recipe and reproduced metrics.

Additional Markdown cells may split dense explanations, but architecture code
must retain the stated component boundaries.

## Full Architecture Visualization

The opening diagram will describe the complete model path:

```text
six images [B,6,3,H,W]
  |
  +-> GridMask during training
  |
  +-> shared Caffe-style ResNet-101 + modulated DCNv2
  |     C3 [B,6,512,H/8,W/8]
  |     C4 [B,6,1024,H/16,W/16]
  |     C5 [B,6,2048,H/32,W/32]
  |
  +-> FPN
  |     P3..P6, each [B,6,256,H_l,W_l]
  |
900 learned query/content embeddings [B,900,256]
900 learned query-position embeddings [B,900,256]
  |
  +-> initial references [B,900,3]
  |
  +-> decoder layer x6
        self-attention
          -> project references into six cameras
          -> sample P3..P6
          -> learned camera/level cross-attention
        FFN
        classification [B,900,10]
        regression [B,900,10]
        detached refined references [B,900,3]
  |
  +-> all layer logits [6,B,900,10]
  +-> all layer boxes  [6,B,900,10]
```

Focused ASCII diagrams will use the same symbols and field order as the code.

## Shape Annotation Standard

Symbols are defined once and restated at section boundaries:

- `B`: batch size
- `N=6`: camera count
- `Q=900`: query count
- `C=256`: embedding channels
- `L=4`: FPN level count
- `K=10`: detection classes
- `D=10`: encoded box dimensions

Every operation that changes rank, axis order, or dimension size receives an
inline comment. The standard form is:

```python
images = images.flatten(0, 1)  # [B,N,3,H,W] -> [B*N,3,H,W]; faithful N=6, H=928, W=1600
features = features.view(B, N, C, H_l, W_l)  # [B*N,C,H_l,W_l] -> [B,N,C,H_l,W_l]; C=256
```

Shape-preserving arithmetic does not require repetitive comments unless it
changes tensor meaning, broadcasting behavior, or coordinate convention.

## Required Equations

Markdown must derive and connect at least these equations to implementation
lines:

- FPN top-down recursion and extra P6 construction
- normalized reference denormalization into point-cloud coordinates
- homogeneous LiDAR-to-image projection
- pixel-to-`grid_sample` coordinate normalization
- camera/level attention weights and masked feature fusion
- self-attention residual and normalization order
- cross-attention residual and position encoding
- FFN residual and normalization order
- inverse-sigmoid iterative center refinement
- semantic 9D to encoded 10D conversion
- encoded 10D to semantic 9D conversion

## Execution Modes

The notebook uses explicit configuration values rather than hidden environment
assumptions:

```text
SMOKE_MODE=True
RUN_REAL_DATA=False
RUN_CHECKPOINT=False
RUN_TRAIN_STEP=False
RUN_OFFICIAL_EVAL=False
```

Default execution uses reduced synthetic image sizes, camera count, query count,
and channel widths where supported. The faithful builder remains separate and
uses the exact C6 dimensions and module layout.

Optional cells must:

- state which local paths or dependencies they require
- skip with a concise reason when disabled or unavailable
- fail directly for malformed paths when explicitly enabled
- avoid silently switching to a different architecture

## Checkpoint Compatibility

The notebook architecture must preserve the production `state_dict` contract.
Validation with `outputs/c6_full_v1/final_checkpoint.pt` must check:

- strict model state loading
- no missing keys
- no unexpected keys
- matching class count and ordering
- matching query, embedding, layer, and FPN dimensions
- matching one-sample evaluation outputs against the production model within
  floating-point tolerance

The notebook may import configuration-resolution and FCOS3D translation helpers,
but the loaded target model must be notebook-defined.

## Validation

### Static Notebook Checks

A test will inspect the notebook and verify:

- no imports of project model classes or `build_model`
- required architecture section markers are present
- each architecture component has a local definition
- no stored cell outputs
- no execution counts
- all code cells parse as Python after notebook magics are excluded

### Smoke Execution

The output-free code cells must execute sequentially in smoke mode and cover:

- GridMask behavior
- backbone stage output contracts
- FPN level contracts
- feature projection, masking, and sampling
- cross-attention fusion
- one decoder layer
- six-layer reference refinement
- final class and box shapes
- 9D/10D box round trip
- one forward/loss/backward/optimizer step using allowed production loss imports

### Component Parity

During implementation, notebook components and package components will receive
identical parameters and deterministic inputs. Tests will compare:

- backbone stage outputs
- FPN outputs
- sampled grids, masks, and features
- cross-attention outputs
- decoder outputs
- head logits, boxes, and references
- top-level six-layer outputs
- box codec outputs

### Repository Verification

The existing repository test suite must continue to pass. The notebook must be
saved output-free after any execution-based verification.

## Error Handling

- Invalid input ranks or incompatible dimensions should raise the same natural
  PyTorch errors as production code unless an explicit assertion makes the
  contract clearer.
- Optional data and checkpoint cells skip only when disabled or assets are
  absent.
- Explicitly enabled cells fail with a clear path or dependency message.
- Strict checkpoint incompatibility is always an error; partial loading is not
  accepted for the faithful parity demonstration.

## Maintenance Rules

- Source comments identify the mirrored production file and symbols without
  importing them.
- Notebook behavior changes must first be justified against production code.
- Intentional notebook-only smoke-size accommodations must not alter the
  faithful builder.
- Production behavior and notebook behavior should be parity-tested at component
  boundaries to detect future drift.
- The committed notebook remains educational and output-free; generated
  execution copies belong outside the repository.
