# Full DETR3D Architecture Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the walkthrough notebook with an output-free, standalone, checkpoint-compatible implementation of the complete faithful DETR3D architecture, with equations, ASCII diagrams, and inline tensor-shape evolution.

**Architecture:** Mirror the model implementation at commit `5244860688cfc390a82f8c730017853188b50e5f` directly into ordered notebook cells. Keep architecture code local to the notebook while allowing production imports for data, loss, training, checkpoint, diagnostics, and official evaluation demonstrations. Validate static structure, smoke execution, component parity, and strict faithful-checkpoint parity.

**Tech Stack:** Python 3.11, Jupyter notebook JSON, PyTorch, torchvision, SciPy, Pillow, NumPy, Matplotlib, pytest.

## Global Constraints

- Do not import from `detr3d.models` or import a project `build_model` in the notebook.
- Define every architecture component listed in the design spec in a local code cell.
- Give each public submodule or major architecture chunk its own code cell with immediately preceding explanatory Markdown.
- Add symbolic and faithful inline shape comments to every rank-, axis-, channel-, or spatial-dimension-changing operation.
- Use the faithful resize target `900x1600` and padded model input `928x1600` in explanations.
- Preserve production class names, attributes, parameter names, initialization, operation order, and `state_dict` keys.
- Default execution uses reduced synthetic inputs and requires no dataset or checkpoint.
- Save the committed notebook with `outputs=[]` and `execution_count=null` in every code cell.
- Do not alter production model behavior to make notebook parity easier.

---

## File Structure

- Modify `notebooks/detr3d_e2e_walkthrough.ipynb`: complete literate architecture implementation and optional integration demonstrations.
- Create `tests/test_architecture_notebook.py`: deterministic notebook structure, independence, shape-comment, smoke-execution, and component-parity checks.
- Modify `README.md`: accurately describe the notebook as a standalone architecture implementation.
- Modify `COMMAND_GUIDE.md`: replace outdated notebook-import wording with smoke and optional parity execution instructions.

### Task 1: Establish The Notebook Contract Test

**Files:**
- Create: `tests/test_architecture_notebook.py`
- Test: `tests/test_architecture_notebook.py`

**Interfaces:**
- Consumes: `notebooks/detr3d_e2e_walkthrough.ipynb` as notebook JSON.
- Produces: `load_notebook()`, `markdown_text()`, `code_text()`, and static contract tests used throughout the plan.

- [ ] **Step 1: Write static contract tests**

Create helpers and tests with these exact required markers and local definitions:

```python
import ast
import json
from pathlib import Path

NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "detr3d_e2e_walkthrough.ipynb"
MARKERS = {
    "ARCH-GRID-MASK",
    "ARCH-BACKBONE",
    "ARCH-FPN",
    "ARCH-REFERENCES",
    "ARCH-FEATURE-SAMPLING",
    "ARCH-CROSS-ATTENTION",
    "ARCH-DECODER-LAYER",
    "ARCH-TRANSFORMER",
    "ARCH-HEAD",
    "ARCH-MODEL",
    "ARCH-BOX-CODEC",
}
DEFINITIONS = {
    "GridMask",
    "ModulatedDeformConv2dPack",
    "MultiViewImageBackbone",
    "ImageFPN",
    "inverse_sigmoid",
    "denormalize_reference_points",
    "feature_sampling",
    "Detr3DCrossAttention",
    "Detr3DDecoderLayer",
    "Detr3DTransformer",
    "MLP",
    "Detr3DHead",
    "Detr3DModel",
    "encode_bbox_targets",
    "decode_bbox_predictions",
    "build_notebook_model",
}

def load_notebook():
    return json.loads(NOTEBOOK.read_text())

def markdown_text(notebook):
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )

def code_text(notebook):
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
```

Tests must assert all markers and definitions exist, no code cell has outputs or an execution count, all code parses after dropping full `%`/`!` lines, and no AST import has a module beginning with `detr3d.models` or imports `build_model` from project modules.

- [ ] **Step 2: Add dimension-operation comment checks**

Parse code cell source lines and require inline bracketed shape comments on lines containing `.view(`, `.reshape(`, `.flatten(`, `.unflatten(`, `.squeeze(`, `.unsqueeze(`, `.permute(`, `.transpose(`, `torch.stack(`, `torch.cat(`, `F.interpolate(`, or `F.grid_sample(`. Exclude comments, docstrings, and multiline continuation lines from the mechanical check; retain a manual-review assertion list for convolution, pooling, attention, and linear call sites.

- [ ] **Step 3: Run the tests and verify failure**

Run:

```bash
conda run -n torch_env env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_architecture_notebook.py -q
```

Expected: failures for missing markers and local definitions in the existing import-based notebook.

- [ ] **Step 4: Commit the failing contract tests**

```bash
git add tests/test_architecture_notebook.py
git commit -m "test: define architecture notebook contract"
```

### Task 2: Build Notebook Foundation, GridMask, Backbone, And FPN

**Files:**
- Modify: `notebooks/detr3d_e2e_walkthrough.ipynb`
- Test: `tests/test_architecture_notebook.py`

**Interfaces:**
- Consumes: PyTorch tensors `[B,N,3,H,W]` and standard torchvision ResNet/DCNv2 building blocks.
- Produces: `GridMask`, `ModulatedDeformConv2dPack`, `MultiViewImageBackbone`, and `ImageFPN`; FPN mapping with `p3` through `p6`, each `[B,N,256,H_l,W_l]` in faithful mode.

- [ ] **Step 1: Replace the notebook introduction and configuration cells**

Add the full architecture ASCII diagram, symbols `B,N,Q,C,L,K,D`, the `900x1600 -> 928x1600` preprocessing contract, deterministic seed setup, and:

```python
SMOKE_MODE = True
RUN_REAL_DATA = False
RUN_CHECKPOINT = False
RUN_TRAIN_STEP = True
RUN_OFFICIAL_EVAL = False
NUSCENES_DATAROOT = None
CHECKPOINT_PATH = None
FCOS3D_CHECKPOINT_PATH = None
```

Use only standard-library, PyTorch, torchvision, NumPy, Pillow, SciPy, and Matplotlib imports in the architecture portion.

- [ ] **Step 2: Add `ARCH-GRID-MASK` Markdown and implementation cells**

Mirror `detr3d/models/grid_mask.py:GridMask`, explain the random grid equations and shared flattened-image mask, and annotate image/mask broadcasting. Add a probe asserting evaluation mode is identity and training mode preserves `[B,N,3,H,W]`.

- [ ] **Step 3: Add `ARCH-BACKBONE` Markdown and DCNv2 cell**

Mirror `ModulatedDeformConv2dPack` from `detr3d/models/backbone/image_backbone.py`. Explain offsets `[BN,2k²,H_o,W_o]`, modulation masks `[BN,k²,H_o,W_o]`, concatenated prediction `[BN,3k²,H_o,W_o]`, and output `[BN,C_o,H_o,W_o]`.

- [ ] **Step 4: Add the multi-view backbone cell**

Mirror `MultiViewImageBackbone`, including Caffe stride conversion, stage 4/5 modulated DCNv2 replacement, frozen stem, BatchNorm evaluation behavior, and camera flatten/restore. Document faithful outputs:

```text
C3 [B,6,512,116,200]
C4 [B,6,1024,58,100]
C5 [B,6,2048,29,50]
```

- [ ] **Step 5: Add `ARCH-FPN` Markdown and implementation cells**

Mirror `ImageFPN` from `detr3d/models/neck/fpn.py`, include equations for lateral projection, top-down interpolation/addition, output convolution, and ReLU-before-P6. Document faithful P3-P6 shapes.

- [ ] **Step 6: Add reduced deterministic probes**

Construct smoke modules with `pretrained=False`, one camera, small image dimensions, and evaluation mode. Assert backbone and FPN ranks, camera-axis restoration, 256 output channels, decreasing spatial sizes, and P6 shape.

- [ ] **Step 7: Run focused static tests**

Run the marker/definition/output tests for GridMask, backbone, and FPN. Expected: those group assertions pass; later architecture-group assertions still fail.

- [ ] **Step 8: Commit the architecture front end**

```bash
git add notebooks/detr3d_e2e_walkthrough.ipynb
git commit -m "docs: implement notebook image feature path"
```

### Task 3: Implement References, Projection, Sampling, And Cross-Attention

**Files:**
- Modify: `notebooks/detr3d_e2e_walkthrough.ipynb`
- Modify: `tests/test_architecture_notebook.py`

**Interfaces:**
- Consumes: FPN list of `[B,N,C,H_l,W_l]`, normalized references `[B,Q,3]`, and image metadata.
- Produces: metric references `[B,Q,3]`, sampled features `[B,C,Q,N,1,L]`, validity masks `[B,1,Q,N,1,L]`, and cross-attention output `[B,Q,C]`.

- [ ] **Step 1: Add `ARCH-REFERENCES` explanation and helpers**

Mirror `inverse_sigmoid` and `denormalize_reference_points`. Include:

```text
p = p_min + r * (p_max - p_min)
```

and annotate `[B,Q,3] -> [B,Q,3]` as a semantic coordinate-space change.

- [ ] **Step 2: Add `ARCH-FEATURE-SAMPLING` derivation and implementation**

Mirror `_stack_projection_matrices`, `_stack_image_shapes`, and `feature_sampling`. Derive homogeneous projection, depth division, pixel normalization to `[-1,1]`, validity rules, camera flattening, `grid_sample`, axis permutation, and level stacking. Every rank or axis change receives an inline shape comment.

- [ ] **Step 3: Add an analytic geometry probe**

Use one identity-like camera, a coordinate-coded feature map, one visible point, one out-of-frame point, and one behind-camera point. Assert projected metric points, expected bilinear sample, masks, and final `[B,C,Q,N,1,L]` layout.

- [ ] **Step 4: Add `ARCH-CROSS-ATTENTION` equations and implementation**

Mirror `Detr3DCrossAttention`, including sigmoid camera/level weights, mask multiplication, camera/level reduction, output projection, dropout residual, inverse-sigmoid reference position encoding, and `reset_parameters` behavior.

- [ ] **Step 5: Add cross-attention probe and parity test**

Instantiate notebook and production components with identical state dictionaries and deterministic tensors. Compare metric references, sampled features, mask, and final cross-attention output with `torch.testing.assert_close`.

- [ ] **Step 6: Run focused tests and commit**

```bash
conda run -n torch_env env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_architecture_notebook.py tests/test_cross_attention.py -q
git add notebooks/detr3d_e2e_walkthrough.ipynb tests/test_architecture_notebook.py
git commit -m "docs: implement notebook multiview attention"
```

### Task 4: Implement Decoder, Transformer, Head, Model, And Box Codec

**Files:**
- Modify: `notebooks/detr3d_e2e_walkthrough.ipynb`
- Modify: `tests/test_architecture_notebook.py`

**Interfaces:**
- Consumes: FPN features, image metadata, query embeddings, and references.
- Produces: `cls_scores [6,B,Q,K]`, `bbox_preds [6,B,Q,10]`, `init_reference [B,Q,3]`, and `inter_references [6,B,Q,3]`.

- [ ] **Step 1: Add `ARCH-DECODER-LAYER` explanation and implementation**

Mirror `Detr3DDecoderLayer` with the exact sequence self-attention, residual, norm, cross-attention, norm, FFN, residual, norm. Explain that cross-attention owns its residual and must not receive an extra wrapper residual.

- [ ] **Step 2: Add `ARCH-TRANSFORMER` explanation and implementation**

Mirror `Detr3DTransformer`, including learned query/content embeddings, learned query-position embeddings, Xavier initialization, cross-attention reset, `init_decoder_state`, and the standalone `forward` method.

- [ ] **Step 3: Add `ARCH-HEAD` equations and `MLP` cell**

Explain the classification and regression branches, output order, initialization, and normalized initial references. Implement `MLP` in its own code cell.

- [ ] **Step 4: Add the `Detr3DHead` cell**

Mirror all production methods, including `_encode_box_predictions`, `init_reference_points`, `regress_boxes`, `classify`, `refine_reference_points_from_reg_output`, `forward_single`, `predict_reference_points`, and `forward`. Derive:

```text
r_next_xy = sigmoid(delta_xy + inverse_sigmoid(r_xy))
r_next_z  = sigmoid(delta_z  + inverse_sigmoid(r_z))
```

and show that z uses regression index 4.

- [ ] **Step 5: Add `ARCH-MODEL` diagram and top-level model**

Mirror `Detr3DModel.forward` as the authoritative six-layer loop. Explicitly show prediction after each decoder layer and detached refined references passed to the next layer.

- [ ] **Step 6: Add the local faithful builder**

Define `build_notebook_model` with the same module nesting, attributes, defaults, and C6 construction as production `build_model`; do not import project model code.

- [ ] **Step 7: Add `ARCH-BOX-CODEC` equations and implementation**

Mirror semantic `[x,y,z,w,l,h,yaw,vx,vy]` to encoded `[x,y,log(w),log(l),z,log(h),sin(yaw),cos(yaw),vx,vy]` and its inverse. Add finite-value, round-trip, and wrapped-yaw assertions.

- [ ] **Step 8: Add reduced end-to-end forward probe**

Use a reduced query count and image size while retaining the same class/module boundaries. Assert all six output layers and verify references remain in `[0,1]` and are detached between layers.

- [ ] **Step 9: Add component parity tests**

Execute notebook definitions into an isolated namespace, instantiate production and notebook components, copy state dictionaries, and compare decoder-layer, transformer, head, top-level model, and codec outputs using deterministic fixtures.

- [ ] **Step 10: Run architecture tests and commit**

```bash
conda run -n torch_env env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_architecture_notebook.py tests/test_decoder_layer.py tests/test_model_forward.py tests/test_initialization.py tests/test_cross_attention.py -q
git add notebooks/detr3d_e2e_walkthrough.ipynb tests/test_architecture_notebook.py
git commit -m "docs: implement complete notebook DETR3D model"
```

### Task 5: Add Training And Evaluation Integration Demonstrations

**Files:**
- Modify: `notebooks/detr3d_e2e_walkthrough.ipynb`
- Modify: `tests/test_architecture_notebook.py`

**Interfaces:**
- Consumes: notebook-defined `Detr3DModel` and allowed production infrastructure.
- Produces: smoke forward/loss/backward/update, optional real data, strict checkpoint parity, diagnostics, and official evaluation cells.

- [ ] **Step 1: Add production data and loss integration cells**

Import only approved non-architecture symbols. Explain the semantic 9D GT to encoded 10D matching/loss contract and demonstrate `Detr3DLoss` consuming notebook-model outputs.

- [ ] **Step 2: Add mandatory smoke training step**

Create deterministic synthetic targets, run forward, all-layer loss, backward, gradient clipping, and one AdamW update. Assert finite loss and nonzero finite gradients. Keep `RUN_TRAIN_STEP=True` in smoke mode.

- [ ] **Step 3: Add optional real-data integration**

When `RUN_REAL_DATA=True`, validate `NUSCENES_DATAROOT`, instantiate one validation sample through production data code, print image/metadata/GT contracts, and feed it into the notebook model. When disabled or absent, print one concise skip reason.

- [ ] **Step 4: Add optional strict checkpoint parity**

When `RUN_CHECKPOINT=True`, require CUDA, dataset, and checkpoint. Build notebook and production models in the same process, strict-load the same checkpoint state, use validation sample 0 in float32/eval mode, and compare `cls_scores`, `bbox_preds`, `init_reference`, and `inter_references` at `atol=rtol=1e-5`.

- [ ] **Step 5: Add diagnostic and official evaluation cells**

Use approved production diagnostics/evaluator imports with notebook-model outputs. Explain diagnostic per-query decoding versus official flattened query-class top-300 decoding. Guard full export/devkit execution behind `RUN_OFFICIAL_EVAL` and validate all required paths.

- [ ] **Step 6: Add training recipe and result cells**

Document the faithful C6 command source, 24-epoch result `34.58` mAP and `42.15` NDS, and the distinction between regression checks and official metrics.

- [ ] **Step 7: Run smoke execution test and commit**

Execute all notebook code cells sequentially in a subprocess with optional modes disabled, then assert no exception and the smoke completion sentinel is printed.

```bash
conda run -n torch_env env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_architecture_notebook.py -q
git add notebooks/detr3d_e2e_walkthrough.ipynb tests/test_architecture_notebook.py
git commit -m "docs: complete notebook training and evaluation flow"
```

### Task 6: Update Documentation And Perform Final Verification

**Files:**
- Modify: `README.md`
- Modify: `COMMAND_GUIDE.md`
- Modify: `notebooks/detr3d_e2e_walkthrough.ipynb`
- Test: `tests/test_architecture_notebook.py`

**Interfaces:**
- Consumes: completed architecture notebook.
- Produces: accurate user instructions and an output-free verified notebook.

- [ ] **Step 1: Update README notebook description**

State that the notebook locally defines the full DETR3D architecture, uses production imports only for surrounding data/training/evaluation demonstrations, runs synthetic smoke mode by default, and offers guarded real-data/checkpoint parity modes.

- [ ] **Step 2: Update command guide**

Document the exact smoke execution command and optional path configuration. Remove claims that the notebook imports the model or trainer as its primary implementation.

- [ ] **Step 3: Clear notebook outputs and execution counts**

Ensure every code cell has:

```json
"execution_count": null,
"outputs": []
```

- [ ] **Step 4: Run notebook-focused verification**

```bash
conda run -n torch_env env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_architecture_notebook.py -q
```

Expected: all notebook contract, parsing, smoke, and component parity tests pass.

- [ ] **Step 5: Run the full repository suite**

```bash
conda run -n torch_env env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

Expected: at least the existing 72 tests plus new notebook tests pass.

- [ ] **Step 6: Run optional local checkpoint parity**

Use `outputs/c6_full_v1/final_checkpoint.pt`, nuScenes validation sample 0, CUDA, float32, and evaluation mode. Expected: strict state loading and all four output tensors pass `torch.testing.assert_close(atol=1e-5, rtol=1e-5)`.

- [ ] **Step 7: Inspect repository hygiene**

Verify `git diff --check`, no notebook outputs, no tracked checkpoints/datasets/logs/MLflow artifacts, and only intended files changed.

- [ ] **Step 8: Commit final documentation and verification updates**

```bash
git add README.md COMMAND_GUIDE.md notebooks/detr3d_e2e_walkthrough.ipynb tests/test_architecture_notebook.py
git commit -m "docs: publish full DETR3D architecture notebook"
```
