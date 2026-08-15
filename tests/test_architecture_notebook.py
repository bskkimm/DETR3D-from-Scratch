import ast
import json
import math
import os
import subprocess
import sys
import tokenize
from io import StringIO
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from PIL import Image
from torchvision import models
from torchvision.models import ResNet101_Weights
from torchvision.ops import DeformConv2d

from detr3d.models.backbone import MultiViewImageBackbone as ProductionMultiViewImageBackbone
from detr3d.models.transformer.cross_attention import (
    Detr3DCrossAttention as ProductionDetr3DCrossAttention,
)
from detr3d.models.transformer.feature_sampling import (
    feature_sampling as production_feature_sampling,
)
from detr3d.models.detr3d import Detr3DModel as ProductionDetr3DModel
from detr3d.models.grid_mask import GridMask as ProductionGridMask
from detr3d.models.heads.detr3d_head import (
    Detr3DHead as ProductionDetr3DHead,
    MLP as ProductionMLP,
)
from detr3d.models.losses.loss_utils import (
    decode_bbox_predictions as production_decode_bbox_predictions,
    encode_bbox_targets as production_encode_bbox_targets,
)
from detr3d.models.neck import ImageFPN as ProductionImageFPN
from detr3d.models.transformer.decoder import (
    Detr3DTransformer as ProductionDetr3DTransformer,
)
from detr3d.models.transformer.decoder_layer import (
    Detr3DDecoderLayer as ProductionDetr3DDecoderLayer,
)


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
IMAGE_FEATURE_MARKERS = {"ARCH-GRID-MASK", "ARCH-BACKBONE", "ARCH-FPN"}
IMAGE_FEATURE_DEFINITIONS = {
    "DeformConv2dPack",
    "GridMask",
    "ModulatedDeformConv2dPack",
    "MultiViewImageBackbone",
    "ImageFPN",
}
MULTIVIEW_ATTENTION_MARKERS = {
    "ARCH-REFERENCES",
    "ARCH-FEATURE-SAMPLING",
    "ARCH-CROSS-ATTENTION",
}
MULTIVIEW_ATTENTION_DEFINITIONS = {
    "inverse_sigmoid",
    "denormalize_reference_points",
    "_stack_projection_matrices",
    "_stack_image_shapes",
    "feature_sampling",
    "Detr3DCrossAttention",
}
TASK4_MARKERS = {
    "ARCH-DECODER-LAYER",
    "ARCH-TRANSFORMER",
    "ARCH-HEAD",
    "ARCH-MODEL",
    "ARCH-BOX-CODEC",
}
TASK4_DEFINITIONS = {
    "Detr3DDecoderLayer",
    "Detr3DTransformer",
    "MLP",
    "Detr3DHead",
    "Detr3DModel",
    "encode_bbox_targets",
    "decode_bbox_predictions",
    "build_notebook_model",
}
APPROVED_PRODUCTION_IMPORTS = {
    ("detr3d.data", "NuScenesDetr3DDataset"),
    ("detr3d.engine.diagnostics", "decode_predictions"),
    ("detr3d.engine.evaluator", "decode_nuscenes_predictions"),
    ("detr3d.engine.evaluator", "export_nuscenes_results"),
    ("detr3d.engine.evaluator", "run_nuscenes_evaluation"),
    ("detr3d.models.losses", "Detr3DLoss"),
    ("detr3d.models.losses.matcher", "HungarianMatcher3D"),
}

DIMENSION_OPERATIONS = (
    ".view(",
    ".reshape(",
    ".flatten(",
    ".unflatten(",
    ".squeeze(",
    ".unsqueeze(",
    ".permute(",
    ".transpose(",
    "torch.stack(",
    "torch.cat(",
    "F.interpolate(",
    "F.grid_sample(",
)

TASK2_DIMENSION_ANNOTATIONS = (
    ("GridMask canvas", "mask = np.ones(", "[1392,2400]"),
    ("GridMask crop", "mask = mask[top :", "[1392,2400] -> [928,1600]"),
    ("GridMask mask view", "mask_tensor = mask_tensor.view(", "[928,1600] -> [1,1,928,1600]"),
    ("GridMask broadcast", "return images * mask_tensor  ", "[B*6,3,928,1600]"),
    ("GridMask noise view", ").view(1, 1, height, width)  #", "[928,1600] -> [1,1,928,1600]"),
    ("GridMask noise broadcast", "return images * mask_tensor + noise", "[B*6,3,928,1600]"),
    ("DCNv1 offset convolution", "offset = self.offset_conv(x)", "[B*6,256,116,200] -> [B*6,18,58,100]"),
    ("DCNv1 output convolution", "return self.deform_conv(x, offset)  ", "[B*6,256,116,200] -> [B*6,256,58,100]"),
    ("DCNv2 offset-mask convolution", "offset_mask = self.conv_offset(x)", "[B*6,256,58,100] -> [B*6,27,58,100]"),
    ("DCNv2 offset split", "offset = offset_mask[:, :", "[B*6,27,58,100] -> [B*6,18,58,100]"),
    ("DCNv2 mask split", "mask = offset_mask[:, self.offset_channels :]", "[B*6,27,58,100] -> [B*6,9,58,100]"),
    ("DCNv2 output convolution", "return self.deform_conv(x, offset, mask)", "[B*6,512,29,50]"),
    ("backbone camera flatten", "flat = images.reshape(", "[B,6,3,928,1600] -> [B*6,3,928,1600]"),
    ("backbone stem convolution and pooling", "x = self.stem(flat)", "[B*6,3,928,1600] -> [B*6,64,232,400]"),
    ("backbone stage 2", "stage2 = self.stage2(x)", "[B*6,64,232,400] -> [B*6,256,232,400]"),
    ("backbone stage 3", "stage3 = self.stage3(stage2)", "[B*6,256,232,400] -> [B*6,512,116,200]"),
    ("backbone stage 4", "stage4 = self.stage4(stage3)", "[B*6,512,116,200] -> [B*6,1024,58,100]"),
    ("backbone stage 5", "stage5 = self.stage5(stage4)", "[B*6,1024,58,100] -> [B*6,2048,29,50]"),
    ("backbone camera restore", "multi_view[name] = feat.reshape(", "C3 [B*6,512,116,200] -> [B,6,512,116,200]"),
    ("FPN camera flatten", "flat = feat.reshape(", "C5 [B,6,2048,29,50] -> [B*6,2048,29,50]"),
    ("FPN lateral convolution", "lateral = conv(feat)", "C3 [B*6,512,116,200] -> [B*6,256,116,200]"),
    ("FPN interpolation", "upsampled = F.interpolate(", "[B*6,256,29,50] -> [B*6,256,58,100]"),
    ("FPN output convolution", "output = conv(feat)", "P5 [B*6,256,29,50] -> [B*6,256,29,50]"),
    ("FPN P6 convolution", "outputs.append(self.extra_conv(extra_input))", "[B*6,256,29,50] -> [B*6,256,15,25]"),
    ("FPN camera restore", "pyramid[out_name] = feat.reshape(", "P6 [B*6,256,15,25] -> [B,6,256,15,25]"),
    ("smoke backbone", "smoke_features = smoke_backbone(smoke_images)", "C5 [B,6,2048,29,50]"),
    ("smoke FPN", "smoke_pyramid = smoke_fpn(smoke_features)", "P6 [B,6,256,15,25]"),
)

TASK3_DIMENSION_ANNOTATIONS = (
    ("point-cloud minimum slice", "xyz_min = pc_range_tensor[:3]", "[6] -> [3]"),
    ("point-cloud maximum slice", "xyz_max = pc_range_tensor[3:]", "[6] -> [3]"),
    ("homogeneous one column", "ref_points_one = torch.ones_like(", "[B,900,3] -> [B,900,1]"),
    ("homogeneous coordinate append", "ref_points_homo = torch.cat(", "[B,900,3] -> [B,900,4]"),
    ("projection matrix batch stack", "return torch.stack(matrices, dim=0)", "[B,6,4,4]"),
    ("image shape camera-axis insertion", "shape_tensor = shape_tensor.unsqueeze(0)", "[2] -> [1,2]"),
    ("image shape camera repetition", "shape_tensor = shape_tensor.repeat(", "[1,2] -> [6,2]"),
    ("image height-width slice", "shapes.append(shape_tensor[:, :2])", "[6,S] -> [6,2]"),
    ("image shape batch stack", "return torch.stack(shapes, dim=0)", "[B,6,2]"),
    ("projection camera-axis insertion", "proj = proj.unsqueeze(1)", "[B,4,4] -> [B,1,4,4]"),
    ("projection camera repetition", "proj = proj.repeat(", "[B,1,4,4] -> [B,6,4,4]"),
    ("point camera-axis insertion", "points = ref_points_homo.unsqueeze(1)", "[B,900,4] -> [B,1,900,4]"),
    ("point vector-axis insertion", "points = points.unsqueeze(-1)", "[B,1,900,4] -> [B,1,900,4,1]"),
    ("projection query-axis insertion", "camera_proj = proj.unsqueeze(2)", "[B,6,4,4] -> [B,6,1,4,4]"),
    ("homogeneous projection multiply", "proj_points = torch.matmul(", "[B,6,1,4,4] @ [B,1,900,4,1] -> [B,6,900,4,1]"),
    ("projected vector squeeze", "proj_points = proj_points.squeeze(-1)", "[B,6,900,4,1] -> [B,6,900,4]"),
    ("projected depth slice", "depth = proj_points[..., 2:3]", "[B,6,900,4] -> [B,6,900,1]"),
    ("projected xy slice", "xy = proj_points[..., :2]", "[B,6,900,4] -> [B,6,900,2]"),
    ("image width slice", "width = image_shapes[..., 1:2]", "[B,6,2] -> [B,6,1]"),
    ("image height slice", "height = image_shapes[..., 0:1]", "[B,6,2] -> [B,6,1]"),
    ("pixel x-axis selection", "pixel_x = xy[..., 0]", "[B,6,900,2] -> [B,6,900]"),
    ("pixel y-axis selection", "pixel_y = xy[..., 1]", "[B,6,900,2] -> [B,6,900]"),
    ("sampling grid stack", "grid = torch.stack(", "[B,6,900] + [B,6,900] -> [B,6,900,2]"),
    ("depth-axis selection", "valid = depth[..., 0]", "[B,6,900,1] -> [B,6,900]"),
    ("grid x-axis selection", "grid_x = grid[..., 0]", "[B,6,900,2] -> [B,6,900]"),
    ("grid y-axis selection", "grid_y = grid[..., 1]", "[B,6,900,2] -> [B,6,900]"),
    ("feature camera flatten", "feat = level_feat.reshape(", "[B,6,256,H_l,W_l] -> [B*6,256,H_l,W_l]"),
    ("grid camera flatten", "level_grid = grid.reshape(", "[B,6,900,2] -> [B*6,900,1,2]"),
    ("bilinear sampling", "sampled = F.grid_sample(", "[B*6,256,H_l,W_l] -> [B*6,256,900,1]"),
    ("sample camera restore", "sampled = sampled.reshape(", "[B*6,256,900,1] -> [B,6,256,900,1]"),
    ("sample axis permutation", "sampled = sampled.permute(", "[B,6,256,900,1] -> [B,256,900,6,1]"),
    ("mask axis permutation", "level_mask = valid.permute(", "[B,6,900] -> [B,900,6]"),
    ("mask channel-axis insertion", "level_mask = level_mask.unsqueeze(1)", "[B,900,6] -> [B,1,900,6]"),
    ("mask sample-axis insertion", "level_mask = level_mask.unsqueeze(4)", "[B,1,900,6] -> [B,1,900,6,1]"),
    ("feature level stack", "sampled_feats = torch.stack(", "[B,256,900,6,1] -> [B,256,900,6,1,4]"),
    ("mask level stack", "mask = torch.stack(", "[B,1,900,6,1] -> [B,1,900,6,1,4]"),
    ("attention camera-level view", "attention_weights = attention_weights.view(", "[B,900,24] -> [B,900,6,1,4]"),
    ("attention axis permutation", "attention_weights = attention_weights.permute(", "[B,900,6,1,4] -> [B,1,900,6,4]"),
    ("attention sample-axis insertion", "attention_weights = attention_weights.unsqueeze(4)", "[B,1,900,6,4] -> [B,1,900,6,1,4]"),
    ("attention feature broadcast", "weighted_feats = sampled_feats * attention_weights", "[B,1,900,6,1,4] -> [B,256,900,6,1,4]"),
    ("attention level reduction", "fused_levels = weighted_feats.sum(dim=-1)", "[B,256,900,6,1,4] -> [B,256,900,6,1]"),
    ("attention singleton reduction", "fused_cameras = fused_levels.sum(dim=-1)", "[B,256,900,6,1] -> [B,256,900,6]"),
    ("attention camera reduction", "fused = fused_cameras.sum(dim=-1)", "[B,256,900,6] -> [B,256,900]"),
    ("fused decoder layout", "fused = fused.permute(", "[B,256,900] -> [B,900,256]"),
)

TASK4_DIMENSION_ANNOTATIONS = (
    ("transformer query singleton", "query = self.query_embed.weight.unsqueeze(0)", "[900,256] -> [1,900,256]"),
    ("transformer query batch expansion", "query = query.expand(", "[1,900,256] -> [B,900,256]"),
    ("transformer position singleton", "query_pos = self.query_pos.weight.unsqueeze(0)", "[900,256] -> [1,900,256]"),
    ("transformer position batch expansion", "query_pos = query_pos.expand(", "[1,900,256] -> [B,900,256]"),
    ("transformer state stack", "stacked_states = torch.stack(intermediate_states)", "6 x [B,900,256] -> [6,B,900,256]"),
    ("transformer reference stack", "stacked_refs = torch.stack(intermediate_refs)", "6 x [B,900,3] -> [6,B,900,3]"),
    ("head encoded regression xy slice", "encoded_reg_xy = reg_output[..., 0:2]", "[B,900,10] -> [B,900,2]"),
    ("head encoded reference xy slice", "encoded_ref_xy_logits = ref_logits[..., 0:2]", "[B,900,3] -> [B,900,2]"),
    ("head encoded regression z slice", "encoded_reg_z = reg_output[..., 4:5]", "[B,900,10] -> [B,900,1]"),
    ("head encoded reference z slice", "encoded_ref_z_logits = ref_logits[..., 2:3]", "[B,900,3] -> [B,900,1]"),
    ("head normalized center append", "normalized_centers = torch.cat(", "[B,900,2] + [B,900,1] -> [B,900,3]"),
    ("head refinement regression xy slice", "refine_reg_xy = reg_output[..., 0:2]", "[B,900,10] -> [B,900,2]"),
    ("head refinement reference xy slice", "refine_ref_xy_logits = ref_logits[..., 0:2]", "[B,900,3] -> [B,900,2]"),
    ("head refinement regression z slice", "refine_reg_z = reg_output[..., 4:5]", "[B,900,10] -> [B,900,1]"),
    ("head refinement reference z slice", "refine_ref_z_logits = ref_logits[..., 2:3]", "[B,900,3] -> [B,900,1]"),
    ("head class stack", "stacked_cls_scores = torch.stack(cls_scores)", "6 x [B,900,10] -> [6,B,900,10]"),
    ("head box stack", "stacked_bbox_preds = torch.stack(bbox_preds)", "6 x [B,900,10] -> [6,B,900,10]"),
    ("head hidden-state iteration", "for layer_idx, layer_hs in enumerate(hs):", "[6,B,900,256] -> [B,900,256]"),
    ("model image camera flatten", "flat_images = images.reshape(", "[B,6,3,928,1600] -> [B*6,3,928,1600]"),
    ("model image camera restore", "images = augmented_flat.reshape_as(images)", "[B*6,3,928,1600] -> [B,6,3,928,1600]"),
    ("model class stack", "model_cls_scores = torch.stack(cls_scores)", "6 x [B,900,10] -> [6,B,900,10]"),
    ("model box stack", "model_bbox_preds = torch.stack(bbox_preds)", "6 x [B,900,10] -> [6,B,900,10]"),
    ("model reference stack", "model_inter_references = torch.stack(inter_references)", "6 x [B,900,3] -> [6,B,900,3]"),
    ("empty box encoding", "return boxes.new_zeros((0, 10))", "[0,9] -> [0,10]"),
    ("box encoding concatenate", "encoded_boxes = torch.cat(", "[S,9] -> [S,10]"),
    ("box decoding concatenate", "decoded_boxes = torch.cat(", "[S,10] -> [S,9]"),
)

TASK4_AST_SHAPE_CALLS = {
    "cat",
    "expand",
    "new_zeros",
    "reshape",
    "reshape_as",
    "stack",
    "unsqueeze",
}
TASK4_NON_TENSOR_INDEX_CONTAINERS = {"mlvl_feats"}
TASK4_SEMANTIC_DIMENSION_ANNOTATIONS = (
    ("decoder self-attention", "self_attended, _ = self.self_attn(", 1, "[B,900,256] -> [B,900,256]"),
    ("decoder cross-attention", "cross = self.cross_attn(", 1, "[B,900,256] -> [B,900,256]"),
    ("decoder FFN", "ff = self.ffn(query)", 1, "[B,900,256] -> [B,900,512] -> [B,900,256]"),
    ("standalone reference prediction", "reference_points = reference_point_predictor(", 1, "[B,900,256] -> [B,900,3]"),
    ("decoder layer calls", "hidden = layer(", 2, "[B,900,256] -> [B,900,256]"),
    ("MLP projection", "return self.net(x)", 1, "[B,900,256] -> [B,900,10]"),
    ("initial reference projection", "return self.reference_points(query_pos)", 1, "[B,900,256] -> [B,900,3]"),
    ("standalone reference projection", "return self.reference_points(layer_q)", 1, "[B,900,256] -> [B,900,3]"),
    ("regression branch", "return self.reg_branches[layer_idx](layer_hs)", 1, "[B,900,256] -> [B,900,10]"),
    ("classification branch", "return self.cls_branches[layer_idx](layer_hs)", 1, "[B,900,256] -> [B,900,10]"),
    ("head regression call", "reg_output = self.regress_boxes(", 1, "[B,900,256] -> [B,900,10]"),
    ("head classification call", "cls_score = self.classify(", 1, "[B,900,256] -> [B,900,10]"),
    ("head box encoding call", "bbox_pred = self._encode_box_predictions(", 1, "[B,900,10] -> [B,900,10]"),
    ("backbone feature extraction", "features = self.backbone(images)", 1, "[B,6,3,928,1600] ->"),
    ("FPN feature projection", "pyramid = self.neck(features)", 1, "-> P3-P6 [B,6,256"),
    ("model initial references", "reference_points = self.head.init_reference_points(", 1, "[B,900,256] -> [B,900,3]"),
    ("model regression call", "reg_output = self.head.regress_boxes(", 1, "[B,900,256] -> [B,900,10]"),
    ("model classification call", "cls_score = self.head.classify(", 1, "[B,900,256] -> [B,900,10]"),
    ("model box encoding call", "bbox_pred = self.head._encode_box_predictions(", 1, "[B,900,10] -> [B,900,10]"),
    ("model reference refinement", "reference_points = self.head.refine_reference_points_from_reg_output(", 1, "[B,900,3] -> [B,900,3]"),
)

# Static matching cannot reliably infer dimensions from arbitrary module names.
MANUAL_REVIEW_ASSERTIONS = (
    "Convolution call sites changing channels or spatial size have shape comments.",
    "Pooling call sites changing spatial size have shape comments.",
    "Attention call sites changing tensor layout have shape comments.",
    "Linear call sites changing the final dimension have shape comments.",
)


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


def _code_cells(notebook):
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _parseable_source(cell):
    return "\n".join(
        line
        for line in "".join(cell["source"]).splitlines()
        if not line.lstrip().startswith(("%", "!"))
    )


def _docstring_lines(tree):
    lines = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = node.body
        if not body:
            continue
        first_statement = body[0]
        if (
            isinstance(first_statement, ast.Expr)
            and isinstance(first_statement.value, ast.Constant)
            and isinstance(first_statement.value.value, str)
        ):
            lines.update(
                range(first_statement.lineno, first_statement.end_lineno + 1)
            )
    return lines


def _continuation_lines(tree):
    lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.end_lineno > node.lineno:
            lines.update(range(node.lineno + 1, node.end_lineno + 1))
    return lines


def _comments_by_line(source):
    return {
        token.start[0]: token.string
        for token in tokenize.generate_tokens(StringIO(source).readline)
        if token.type == tokenize.COMMENT
    }


def _declared_markers(markdown):
    return {line.strip() for line in markdown.splitlines()} & MARKERS


def _local_definitions(notebook):
    defined = set()
    for cell in _code_cells(notebook):
        tree = ast.parse(_parseable_source(cell))
        defined.update(
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        )
    return defined


def _execute_multiview_attention_cells():
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
    }
    wanted = MULTIVIEW_ATTENTION_DEFINITIONS
    for cell in _code_cells(load_notebook()):
        source = _parseable_source(cell)
        tree = ast.parse(source)
        names = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if names & wanted:
            exec(compile(source, NOTEBOOK.name, "exec"), namespace)
    return namespace


def _execute_task4_definition_cells():
    namespace = {
        "np": np,
        "math": math,
        "torch": torch,
        "nn": nn,
        "F": F,
        "Image": Image,
        "models": models,
        "ResNet101_Weights": ResNet101_Weights,
        "DeformConv2d": DeformConv2d,
        "Callable": Callable,
        "Dict": Dict,
        "Iterable": Iterable,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
    }
    wanted = MULTIVIEW_ATTENTION_DEFINITIONS | TASK4_DEFINITIONS | {
        "GridMask",
        "ModulatedDeformConv2dPack",
        "MultiViewImageBackbone",
        "ImageFPN",
    }
    for cell in _code_cells(load_notebook()):
        source = _parseable_source(cell)
        tree = ast.parse(source)
        names = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if names & wanted:
            exec(compile(source, NOTEBOOK.name, "exec"), namespace)
    return namespace


def _task4_implementation_sources():
    sources = []
    for cell_index, cell in enumerate(_code_cells(load_notebook())):
        source = _parseable_source(cell)
        tree = ast.parse(source)
        names = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if names & TASK4_DEFINITIONS:
            sources.append((cell_index, source, tree))
    return sources


def _task4_ast_transition_sites(tree):
    sites = {}
    categories = set()
    annotation_node_ids = set()
    for node in ast.walk(tree):
        annotations = []
        if isinstance(node, ast.arg) and node.annotation is not None:
            annotations.append(node.annotation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                annotations.append(node.returns)
        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
        for annotation in annotations:
            annotation_node_ids.update(id(child) for child in ast.walk(annotation))

    def record(line_number, category):
        sites[line_number] = sites.get(line_number, 0) + 1
        categories.add(category)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                operation = node.func.attr
            elif isinstance(node.func, ast.Name):
                operation = node.func.id
            else:
                operation = None
            if operation in TASK4_AST_SHAPE_CALLS:
                record(node.lineno, f"call:{operation}")
        elif isinstance(node, ast.Subscript):
            if (
                id(node) not in annotation_node_ids
                and isinstance(node.value, ast.Name)
                and node.value.id not in TASK4_NON_TENSOR_INDEX_CONTAINERS
            ):
                record(node.lineno, "tensor-index")
        elif isinstance(node, ast.For):
            iterator = node.iter
            if (
                isinstance(iterator, ast.Call)
                and isinstance(iterator.func, ast.Name)
                and iterator.func.id == "enumerate"
                and iterator.args
                and isinstance(iterator.args[0], ast.Name)
            ):
                record(node.lineno, "tensor-iteration")
    return sites, categories


def _forbidden_imports(tree):
    forbidden = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_name = alias.name.rsplit(".", 1)[-1]
                is_project_module = alias.name == "detr3d" or alias.name.startswith(
                    "detr3d."
                )
                is_model_module = alias.name == "detr3d.models" or alias.name.startswith(
                    "detr3d.models."
                )
                if (
                    is_project_module
                    or is_model_module
                    or imported_name in DEFINITIONS
                    or alias.asname in DEFINITIONS
                ):
                    forbidden.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            is_project_module = module == "detr3d" or module.startswith("detr3d.")
            is_model_module = module == "detr3d.models" or module.startswith(
                "detr3d.models."
            )
            for alias in node.names:
                imported_symbol = f"{module}.{alias.name}"
                is_approved = (module, alias.name) in APPROVED_PRODUCTION_IMPORTS
                if (
                    alias.name == "*"
                    or (is_project_module and not is_approved)
                    or (is_model_module and not is_approved)
                    or alias.name in DEFINITIONS
                    or alias.asname in DEFINITIONS
                    or alias.name == "build_model"
                ):
                    forbidden.append(imported_symbol)
    return forbidden


def test_import_contract_allows_production_matching_and_loss_imports():
    tree = ast.parse(
        "from detr3d.models.losses import Detr3DLoss as Loss\n"
        "from detr3d.models.losses.matcher import HungarianMatcher3D as Matcher\n"
    )
    assert _forbidden_imports(tree) == []


def test_import_contract_allows_only_approved_integration_imports():
    tree = ast.parse(
        "from detr3d.data import NuScenesDetr3DDataset\n"
        "from detr3d.engine.diagnostics import decode_predictions\n"
        "from detr3d.engine.evaluator import (\n"
        "    decode_nuscenes_predictions, export_nuscenes_results, run_nuscenes_evaluation\n"
        ")\n"
    )
    assert _forbidden_imports(tree) == []


def test_import_contract_rejects_unapproved_production_infrastructure():
    tree = ast.parse(
        "import detr3d.data as data\n"
        "from detr3d.data import unapproved_helper\n"
        "from detr3d.engine.trainer import fit\n"
    )
    assert _forbidden_imports(tree) == [
        "detr3d.data",
        "detr3d.data.unapproved_helper",
        "detr3d.engine.trainer.fit",
    ]


def test_import_contract_rejects_wildcard_imports_from_every_module():
    tree = ast.parse(
        "from unrelated_package import *\n"
        "from detr3d.models.losses import *\n"
    )
    assert _forbidden_imports(tree) == [
        "unrelated_package.*",
        "detr3d.models.losses.*",
    ]


def test_import_contract_rejects_module_qualified_loss_imports():
    tree = ast.parse(
        "import detr3d.models.losses as losses\n"
        "import detr3d.models.losses.loss_utils as loss_utils\n"
    )
    assert _forbidden_imports(tree) == [
        "detr3d.models.losses",
        "detr3d.models.losses.loss_utils",
    ]


def test_import_contract_rejects_architecture_modules_and_symbols():
    tree = ast.parse(
        "from anywhere import GridMask\n"
        "from detr3d.models.losses.loss_utils import encode_bbox_targets\n"
        "from detr3d.models.backbone.image_backbone import helper\n"
    )
    assert _forbidden_imports(tree) == [
        "anywhere.GridMask",
        "detr3d.models.losses.loss_utils.encode_bbox_targets",
        "detr3d.models.backbone.image_backbone.helper",
    ]


def test_import_contract_rejects_build_model_from_every_module():
    tree = ast.parse("from unrelated_package.factory import build_model\n")
    assert _forbidden_imports(tree) == ["unrelated_package.factory.build_model"]


def test_marker_declarations_must_be_exact_standalone_lines():
    markdown = "ARCH-MODEL-OLD\nprefix ARCH-FPN\nARCH-HEAD\n"
    assert _declared_markers(markdown) == {"ARCH-HEAD"}


def test_required_architecture_markers_are_present():
    markdown = markdown_text(load_notebook())
    missing = MARKERS - _declared_markers(markdown)
    assert not missing, f"missing architecture markers: {sorted(missing)}"


def test_image_feature_architecture_markers_are_present():
    markdown = markdown_text(load_notebook())
    missing = IMAGE_FEATURE_MARKERS - _declared_markers(markdown)
    assert not missing, f"missing image-feature markers: {sorted(missing)}"


def test_required_architecture_definitions_are_local():
    notebook = load_notebook()
    missing = DEFINITIONS - _local_definitions(notebook)
    assert not missing, f"missing local architecture definitions: {sorted(missing)}"


def test_image_feature_architecture_definitions_are_local():
    missing = IMAGE_FEATURE_DEFINITIONS - _local_definitions(load_notebook())
    assert not missing, f"missing local image-feature definitions: {sorted(missing)}"


def test_multiview_attention_contract_is_local_and_complete():
    notebook = load_notebook()
    missing_markers = MULTIVIEW_ATTENTION_MARKERS - _declared_markers(
        markdown_text(notebook)
    )
    missing_definitions = MULTIVIEW_ATTENTION_DEFINITIONS - _local_definitions(notebook)
    assert not missing_markers, f"missing Task 3 markers: {sorted(missing_markers)}"
    assert not missing_definitions, (
        f"missing Task 3 definitions: {sorted(missing_definitions)}"
    )


def test_notebook_feature_sampling_has_hand_checked_projection_geometry():
    notebook = _execute_multiview_attention_cells()
    feature_sampling = notebook["feature_sampling"]
    feature_map = torch.tensor(
        [[[[[0.0, 1.0, 2.0, 3.0],
             [10.0, 11.0, 12.0, 13.0],
             [20.0, 21.0, 22.0, 23.0],
             [30.0, 31.0, 32.0, 33.0]]]]]
    )
    references = torch.tensor(
        [[[0.2, 0.2, 0.75], [0.5, 0.2, 0.75], [0.1, 0.1, 0.25]]]
    )
    metadata = [{"lidar2img": torch.eye(4)[None], "image_shape": (4, 4)}]

    metric, sampled, mask = feature_sampling(
        [feature_map], references, (0.0, 0.0, -2.0, 10.0, 10.0, 2.0), metadata
    )

    torch.testing.assert_close(
        metric, torch.tensor([[[2.0, 2.0, 1.0], [5.0, 2.0, 1.0], [1.0, 1.0, -1.0]]])
    )
    assert sampled.shape == (1, 1, 3, 1, 1, 1)
    torch.testing.assert_close(sampled[0, 0, :, 0, 0, 0], torch.tensor([16.5, 0.0, 0.0]))
    assert mask.shape == (1, 1, 3, 1, 1, 1)
    torch.testing.assert_close(mask[0, 0, :, 0, 0, 0], torch.tensor([1.0, 0.0, 0.0]))


def test_notebook_sampling_and_cross_attention_match_pinned_production():
    notebook = _execute_multiview_attention_cells()
    notebook_feature_sampling = notebook["feature_sampling"]
    NotebookDetr3DCrossAttention = notebook["Detr3DCrossAttention"]
    generator = torch.Generator().manual_seed(31)
    features = [
        torch.randn(1, 2, 4, 4, 5, generator=generator),
        torch.randn(1, 2, 4, 2, 3, generator=generator),
    ]
    references = torch.tensor(
        [[[0.45, 0.55, 0.75], [0.6, 0.4, 0.8], [0.2, 0.3, 0.25]]]
    )
    projections = torch.stack(
        [torch.eye(4), torch.tensor([[1.0, 0.0, 0.0, 0.25], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])]
    )
    metadata = [{"lidar2img": projections, "image_shape": [(4, 5), (4, 5)]}]
    pc_range = (-2.0, -2.0, -1.0, 2.0, 2.0, 3.0)

    expected_sampling = production_feature_sampling(
        features, references, pc_range, metadata
    )
    actual_sampling = notebook_feature_sampling(
        features, references, pc_range, metadata
    )
    for actual, expected in zip(actual_sampling, expected_sampling):
        torch.testing.assert_close(actual, expected)

    torch.manual_seed(41)
    production_attention = ProductionDetr3DCrossAttention(
        embed_dims=4, num_cams=2, num_levels=2, pc_range=pc_range, dropout=0.0
    ).eval()
    torch.manual_seed(41)
    notebook_attention = NotebookDetr3DCrossAttention(
        embed_dims=4, num_cams=2, num_levels=2, pc_range=pc_range, dropout=0.0
    ).eval()
    for name, expected in production_attention.state_dict().items():
        torch.testing.assert_close(notebook_attention.state_dict()[name], expected)
    notebook_attention.load_state_dict(production_attention.state_dict(), strict=True)
    query = torch.randn(1, 3, 4, generator=generator)
    query_pos = torch.randn(1, 3, 4, generator=generator)

    with torch.inference_mode():
        expected_output = production_attention(
            query, features, references, metadata, query_pos
        )
        actual_output = notebook_attention(
            query, features, references, metadata, query_pos
        )
    torch.testing.assert_close(actual_output, expected_output)


def _tiny_decoder_inputs(generator):
    features = [torch.randn(1, 1, 4, 3, 3, generator=generator)]
    query = torch.randn(1, 3, 4, generator=generator)
    query_pos = torch.randn(1, 3, 4, generator=generator)
    references = torch.tensor(
        [[[0.45, 0.55, 0.5], [0.6, 0.4, 0.75], [0.2, 0.3, 0.25]]]
    )
    metadata = [{"lidar2img": torch.eye(4)[None], "image_shape": (3, 3)}]
    return features, query, query_pos, references, metadata


def _assert_matching_state_dicts(actual, expected):
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for name, tensor in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], tensor)


def test_task4_architecture_contract_is_local_and_complete():
    notebook = load_notebook()
    missing_markers = TASK4_MARKERS - _declared_markers(markdown_text(notebook))
    missing_definitions = TASK4_DEFINITIONS - _local_definitions(notebook)
    assert not missing_markers, f"missing Task 4 markers: {sorted(missing_markers)}"
    assert not missing_definitions, (
        f"missing Task 4 definitions: {sorted(missing_definitions)}"
    )


def test_notebook_decoder_layer_matches_pinned_production():
    notebook = _execute_task4_definition_cells()
    NotebookDecoderLayer = notebook["Detr3DDecoderLayer"]
    kwargs = dict(
        embed_dims=4,
        num_heads=1,
        num_cams=1,
        num_levels=1,
        ffn_channels=8,
        dropout=0.0,
        pc_range=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
    )
    torch.manual_seed(101)
    production = ProductionDetr3DDecoderLayer(**kwargs).eval()
    torch.manual_seed(101)
    actual = NotebookDecoderLayer(**kwargs).eval()
    _assert_matching_state_dicts(actual, production)
    actual.load_state_dict(production.state_dict(), strict=True)
    inputs = _tiny_decoder_inputs(torch.Generator().manual_seed(102))

    with torch.inference_mode():
        expected_output = production(
            query=inputs[1], mlvl_feats=inputs[0], query_pos=inputs[2],
            reference_points=inputs[3], img_metas=inputs[4]
        )
        actual_output = actual(
            query=inputs[1], mlvl_feats=inputs[0], query_pos=inputs[2],
            reference_points=inputs[3], img_metas=inputs[4]
        )
    torch.testing.assert_close(actual_output, expected_output)


def test_notebook_transformer_standalone_forward_matches_pinned_production():
    notebook = _execute_task4_definition_cells()
    NotebookTransformer = notebook["Detr3DTransformer"]
    kwargs = dict(
        embed_dims=4,
        num_queries=3,
        num_layers=2,
        num_heads=1,
        num_cams=1,
        num_levels=1,
        pc_range=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
    )
    torch.manual_seed(111)
    production = ProductionDetr3DTransformer(**kwargs).eval()
    torch.manual_seed(111)
    actual = NotebookTransformer(**kwargs).eval()
    _assert_matching_state_dicts(actual, production)
    actual.load_state_dict(production.state_dict(), strict=True)
    features, _, _, _, metadata = _tiny_decoder_inputs(
        torch.Generator().manual_seed(112)
    )
    pyramid = {"p3": features[0]}
    predictor = lambda layer_idx, hidden: torch.sigmoid(hidden[..., :3] + layer_idx)

    with torch.inference_mode():
        expected = production(pyramid, metadata, predictor)
        output = actual(pyramid, metadata, predictor)
    for actual_tensor, expected_tensor in zip(output, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)


def test_notebook_mlp_and_head_match_pinned_production():
    notebook = _execute_task4_definition_cells()
    NotebookMLP = notebook["MLP"]
    NotebookHead = notebook["Detr3DHead"]
    torch.manual_seed(121)
    production_mlp = ProductionMLP(4, 6, 2, 3, use_layernorm=True).eval()
    torch.manual_seed(121)
    actual_mlp = NotebookMLP(4, 6, 2, 3, use_layernorm=True).eval()
    _assert_matching_state_dicts(actual_mlp, production_mlp)

    kwargs = dict(
        embed_dims=4,
        num_classes=3,
        box_dim=10,
        num_decoder_layers=2,
        pc_range=(-2.0, -3.0, -1.0, 2.0, 3.0, 5.0),
    )
    torch.manual_seed(122)
    production = ProductionDetr3DHead(**kwargs).eval()
    torch.manual_seed(122)
    actual = NotebookHead(**kwargs).eval()
    _assert_matching_state_dicts(actual, production)
    actual.load_state_dict(production.state_dict(), strict=True)
    generator = torch.Generator().manual_seed(123)
    hidden = torch.randn(2, 1, 3, 4, generator=generator)
    references = torch.rand(2, 1, 3, 3, generator=generator).clamp(0.1, 0.9)

    with torch.inference_mode():
        expected = production(hidden, references)
        output = actual(hidden, references)
        expected_refined = production.refine_reference_points_from_reg_output(
            production.regress_boxes(0, hidden[0]), references[0]
        )
        actual_refined = actual.refine_reference_points_from_reg_output(
            actual.regress_boxes(0, hidden[0]), references[0]
        )
    for actual_tensor, expected_tensor in zip(output, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)
    torch.testing.assert_close(actual_refined, expected_refined)


class _TinyBackbone(nn.Module):
    def forward(self, images):
        return {"c3": images}


class _TinyNeck(nn.Module):
    def forward(self, features):
        return {"p3": features["c3"]}


def test_notebook_top_level_model_matches_production_and_detaches_refinements():
    notebook = _execute_task4_definition_cells()
    NotebookTransformer = notebook["Detr3DTransformer"]
    NotebookHead = notebook["Detr3DHead"]
    NotebookModel = notebook["Detr3DModel"]
    transformer_kwargs = dict(
        embed_dims=4,
        num_queries=3,
        num_layers=6,
        num_heads=1,
        num_cams=1,
        num_levels=1,
        pc_range=(-1.0, -1.0, 1.0, 1.0, 1.0, 3.0),
    )
    head_kwargs = dict(
        embed_dims=4,
        num_classes=3,
        num_decoder_layers=6,
        pc_range=transformer_kwargs["pc_range"],
    )
    torch.manual_seed(131)
    production_transformer = ProductionDetr3DTransformer(**transformer_kwargs).eval()
    production_head = ProductionDetr3DHead(**head_kwargs).eval()
    torch.manual_seed(131)
    actual_transformer = NotebookTransformer(**transformer_kwargs).eval()
    actual_head = NotebookHead(**head_kwargs).eval()
    actual_transformer.load_state_dict(production_transformer.state_dict(), strict=True)
    actual_head.load_state_dict(production_head.state_dict(), strict=True)
    production = ProductionDetr3DModel(
        _TinyBackbone(), _TinyNeck(), production_transformer, production_head
    ).eval()
    actual = NotebookModel(
        _TinyBackbone(), _TinyNeck(), actual_transformer, actual_head
    ).eval()
    seen_requires_grad = []
    hooks = [
        layer.register_forward_pre_hook(
            lambda module, args, kwargs: seen_requires_grad.append(
                kwargs["reference_points"].requires_grad
            ),
            with_kwargs=True,
        )
        for layer in actual.transformer.layers
    ]
    images = torch.randn(1, 1, 4, 3, 3, generator=torch.Generator().manual_seed(132))
    metadata = [{"lidar2img": torch.eye(4)[None], "image_shape": (3, 3)}]

    expected = production(images, metadata)
    output = actual(images, metadata)
    for hook in hooks:
        hook.remove()
    assert output.keys() == expected.keys()
    for name in output:
        torch.testing.assert_close(output[name], expected[name])
    assert output["cls_scores"].shape == (6, 1, 3, 3)
    assert output["bbox_preds"].shape == (6, 1, 3, 10)
    assert output["init_reference"].shape == (1, 3, 3)
    assert output["inter_references"].shape == (6, 1, 3, 3)
    assert seen_requires_grad == [True, False, False, False, False, False]
    assert torch.all((output["inter_references"] >= 0) & (output["inter_references"] <= 1))


def test_notebook_builder_matches_production_c6_wiring():
    notebook = _execute_task4_definition_cells()
    calls = []

    def record(name):
        def factory(*args, **kwargs):
            calls.append((name, args, kwargs))
            return name
        return factory

    for name in (
        "Detr3DModel",
        "MultiViewImageBackbone",
        "ImageFPN",
        "Detr3DTransformer",
        "Detr3DHead",
        "GridMask",
    ):
        notebook[name] = record(name)

    built = notebook["build_notebook_model"](
        num_queries=900,
        backbone_name="resnet101",
        pretrained_backbone=False,
        official_image_backbone=True,
        use_grid_mask=True,
    )
    assert built == "Detr3DModel"
    assert calls == [
        ("MultiViewImageBackbone", (), {"variant": "resnet101", "pretrained": False, "official_style": True}),
        ("ImageFPN", (), {"relu_before_extra_convs": True}),
        ("Detr3DTransformer", (), {"num_queries": 900, "num_levels": 4}),
        ("Detr3DHead", (), {"num_decoder_layers": 6}),
        ("GridMask", (), {}),
        ("Detr3DModel", (), {
            "backbone": "MultiViewImageBackbone",
            "neck": "ImageFPN",
            "transformer": "Detr3DTransformer",
            "head": "Detr3DHead",
            "image_augmentation": "GridMask",
        }),
    ]


def test_notebook_box_codec_matches_production_and_round_trips_wrapped_yaw():
    notebook = _execute_task4_definition_cells()
    encode = notebook["encode_bbox_targets"]
    decode = notebook["decode_bbox_predictions"]
    boxes = torch.tensor(
        [
            [1.0, -2.0, 0.5, 2.0, 4.0, 1.5, math.pi + 0.25, 0.1, -0.2],
            [-3.0, 1.0, 2.5, 0.5, 1.25, 3.0, -math.pi - 0.5, -1.0, 2.0],
        ]
    )
    expected_encoded = production_encode_bbox_targets(boxes)
    encoded = encode(boxes)
    torch.testing.assert_close(encoded, expected_encoded)
    assert torch.isfinite(encoded).all()
    decoded = decode(encoded)
    torch.testing.assert_close(decoded, production_decode_bbox_predictions(encoded))
    torch.testing.assert_close(decoded[..., :6], boxes[..., :6])
    torch.testing.assert_close(decoded[..., 7:], boxes[..., 7:])
    yaw_error = torch.atan2(
        torch.sin(decoded[..., 6] - boxes[..., 6]),
        torch.cos(decoded[..., 6] - boxes[..., 6]),
    )
    torch.testing.assert_close(yaw_error, torch.zeros_like(yaw_error), atol=1e-6, rtol=0)


def test_code_cells_are_committed_without_outputs():
    for index, cell in enumerate(_code_cells(load_notebook())):
        assert cell.get("outputs") == [], f"code cell {index} has stored outputs"
        assert cell.get("execution_count") is None, (
            f"code cell {index} has an execution count"
        )


def test_all_code_cells_parse_as_python():
    for index, cell in enumerate(_code_cells(load_notebook())):
        try:
            ast.parse(_parseable_source(cell))
        except SyntaxError as error:
            raise AssertionError(f"code cell {index} does not parse: {error}") from error


def test_all_code_cells_execute_sequentially():
    namespace = {}
    for index, cell in enumerate(_code_cells(load_notebook())):
        source = _parseable_source(cell)
        exec(compile(source, f"notebook-cell-{index}", "exec"), namespace)


def test_all_code_cells_smoke_in_fresh_subprocess():
    script = """
import ast
import json
from pathlib import Path

notebook = json.loads(Path('notebooks/detr3d_e2e_walkthrough.ipynb').read_text())
namespace = {}
for index, cell in enumerate(notebook['cells']):
    if cell['cell_type'] != 'code':
        continue
    source = '\\n'.join(
        line
        for line in ''.join(cell['source']).splitlines()
        if not line.lstrip().startswith(('%', '!'))
    )
    ast.parse(source)
    exec(compile(source, f'notebook-cell-{index}', 'exec'), namespace)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=NOTEBOOK.parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "TASK5_SMOKE_COMPLETE" in result.stdout


def test_smoke_training_rejects_a_missing_decoder_layer_loss():
    namespace = {}
    for index, cell in enumerate(_code_cells(load_notebook())):
        source = _parseable_source(cell)
        if "smoke_criterion = Detr3DLoss" not in source:
            exec(compile(source, f"notebook-cell-{index}", "exec"), namespace)
            continue

        real_loss = namespace["Detr3DLoss"]

        class MissingLayerLoss(real_loss):
            def loss_by_feat(self, *args, **kwargs):
                losses = super().loss_by_feat(*args, **kwargs)
                losses.pop("d4.loss_bbox")
                return losses

        namespace["Detr3DLoss"] = MissingLayerLoss
        with pytest.raises(AssertionError, match="all six decoder layers"):
            exec(compile(source, f"notebook-cell-{index}", "exec"), namespace)
        return
    raise AssertionError("smoke training cell was not found")


@pytest.mark.skipif(
    os.environ.get("RUN_NOTEBOOK_REAL_DATA") != "1",
    reason="set RUN_NOTEBOOK_REAL_DATA=1 for the nuScenes notebook fixture",
)
def test_notebook_real_data_path_on_configured_validation_sample(capsys):
    dataroot_value = os.environ.get("NUSCENES_DATAROOT")
    version = os.environ.get("NUSCENES_VERSION")
    sample_index_value = os.environ.get("NUSCENES_SAMPLE_INDEX")
    assert dataroot_value, "RUN_NOTEBOOK_REAL_DATA=1 requires NUSCENES_DATAROOT"
    assert version, "RUN_NOTEBOOK_REAL_DATA=1 requires NUSCENES_VERSION"
    assert sample_index_value is not None, (
        "RUN_NOTEBOOK_REAL_DATA=1 requires NUSCENES_SAMPLE_INDEX"
    )
    dataroot = Path(dataroot_value)
    assert dataroot.is_dir(), (
        f"RUN_NOTEBOOK_REAL_DATA=1 requires a nuScenes dataroot: {dataroot}"
    )
    sample_index = int(sample_index_value)
    assert sample_index >= 0, "NUSCENES_SAMPLE_INDEX must be non-negative"

    namespace = {}
    real_data_cell_executed = False
    for index, cell in enumerate(_code_cells(load_notebook())):
        source = _parseable_source(cell)
        if "real_dataset = None" not in source:
            exec(compile(source, f"notebook-cell-{index}", "exec"), namespace)
            continue
        namespace.update(
            RUN_REAL_DATA=True,
            NUSCENES_DATAROOT=dataroot,
            NUSCENES_VERSION=version,
            NUSCENES_SAMPLE_INDEX=sample_index,
        )
        exec(compile(source, f"notebook-cell-{index}", "exec"), namespace)
        real_data_cell_executed = True
        break

    assert real_data_cell_executed, "notebook real-data cell was not found"
    assert namespace["RUN_REAL_DATA"] is True
    assert namespace["real_dataset"].version == version
    assert len(namespace["real_dataset"]) == sample_index + 1
    assert (
        namespace["real_sample"]["img_metas"]["sample_token"]
        == namespace["real_dataset"].samples[sample_index]["token"]
    )
    assert "TASK5_REAL_DATA_COMPLETE" in capsys.readouterr().out


@pytest.mark.skipif(
    os.environ.get("RUN_NOTEBOOK_CHECKPOINT_PARITY") != "1",
    reason="set RUN_NOTEBOOK_CHECKPOINT_PARITY=1 for the CUDA checkpoint fixture",
)
def test_notebook_strict_checkpoint_parity_on_validation_sample_zero(capsys):
    assert torch.cuda.is_available(), (
        "RUN_NOTEBOOK_CHECKPOINT_PARITY=1 requires CUDA"
    )
    dataroot = Path(
        os.environ.get("NUSCENES_DATAROOT", "/home/beomseokkim2/dataset/nuscenes")
    )
    checkpoint_path = Path(
        os.environ.get(
            "DETR3D_CHECKPOINT_PATH",
            "/home/beomseokkim2/kim/detr3d/outputs/c6_full_v1/final_checkpoint.pt",
        )
    )
    assert dataroot.is_dir(), (
        f"RUN_NOTEBOOK_CHECKPOINT_PARITY=1 requires a nuScenes dataroot: {dataroot}"
    )
    assert checkpoint_path.is_file(), (
        "RUN_NOTEBOOK_CHECKPOINT_PARITY=1 requires a checkpoint file: "
        f"{checkpoint_path}"
    )

    def production_model_factory(config):
        return ProductionDetr3DModel(
            backbone=ProductionMultiViewImageBackbone(
                variant=config["backbone"],
                pretrained=not config["disable_pretrained_backbone"],
                official_style=config["official_image_backbone"],
            ),
            neck=ProductionImageFPN(
                relu_before_extra_convs=config["official_image_backbone"]
            ),
            transformer=ProductionDetr3DTransformer(
                num_queries=config["num_queries"]
            ),
            head=ProductionDetr3DHead(num_decoder_layers=6),
            image_augmentation=(
                ProductionGridMask() if config["grid_mask"] else None
            ),
        )

    output_keys = (
        "cls_scores",
        "bbox_preds",
        "init_reference",
        "inter_references",
    )
    compared_keys = []
    namespace = {}
    parity_cell_executed = False
    for index, cell in enumerate(_code_cells(load_notebook())):
        source = _parseable_source(cell)
        if "checkpoint_notebook_model = None" not in source:
            exec(compile(source, f"notebook-cell-{index}", "exec"), namespace)
            continue

        namespace.update(
            RUN_CHECKPOINT=True,
            NUSCENES_DATAROOT=dataroot,
            CHECKPOINT_PATH=checkpoint_path,
            PRODUCTION_MODEL_FACTORY=production_model_factory,
        )
        real_assert_close = torch.testing.assert_close

        def recording_assert_close(actual, expected, *args, **kwargs):
            notebook_outputs = namespace.get("checkpoint_notebook_outputs", {})
            for key in output_keys:
                if actual is notebook_outputs.get(key):
                    compared_keys.append(key)
                    break
            return real_assert_close(actual, expected, *args, **kwargs)

        torch.testing.assert_close = recording_assert_close
        try:
            exec(compile(source, f"notebook-cell-{index}", "exec"), namespace)
        finally:
            torch.testing.assert_close = real_assert_close
        parity_cell_executed = True
        break

    assert parity_cell_executed, "notebook checkpoint parity cell was not found"
    assert "TASK5_CHECKPOINT_PARITY_COMPLETE" in capsys.readouterr().out
    assert compared_keys == list(output_keys)


def test_notebook_does_not_import_project_architecture():
    forbidden = []
    for cell in _code_cells(load_notebook()):
        forbidden.extend(_forbidden_imports(ast.parse(_parseable_source(cell))))
    assert not forbidden, f"forbidden architecture imports: {sorted(forbidden)}"


def test_explicit_dimension_operations_have_shape_comments():
    missing_comments = []
    for cell_index, cell in enumerate(_code_cells(load_notebook())):
        source = _parseable_source(cell)
        tree = ast.parse(source)
        excluded_lines = _docstring_lines(tree) | _continuation_lines(tree)
        comments = _comments_by_line(source)
        for line_number, line in enumerate(source.splitlines(), start=1):
            if line_number in excluded_lines or line.lstrip().startswith("#"):
                continue
            if not any(operation in line for operation in DIMENSION_OPERATIONS):
                continue
            comment = comments.get(line_number, "")
            if "[" not in comment or "]" not in comment:
                missing_comments.append(f"cell {cell_index}, line {line_number}: {line}")
    assert not missing_comments, "missing bracketed shape comments:\n" + "\n".join(
        missing_comments
    )


def test_task2_dimension_changes_have_symbolic_and_faithful_shape_comments():
    lines = code_text(load_notebook()).splitlines()
    failures = []
    for category, operation, faithful_shape in TASK2_DIMENSION_ANNOTATIONS:
        matches = [line for line in lines if operation in line]
        if len(matches) != 1:
            failures.append(f"{category}: expected one `{operation}` line, found {len(matches)}")
            continue
        comment = matches[0].partition("#")[2]
        required = ("symbolic", "faithful", "->", faithful_shape)
        missing = [fragment for fragment in required if fragment not in comment]
        if missing:
            failures.append(f"{category}: missing {missing}: {matches[0]}")
    assert not failures, "incomplete Task 2 shape annotations:\n" + "\n".join(failures)


def test_task3_dimension_changes_have_symbolic_and_faithful_shape_comments():
    lines = code_text(load_notebook()).splitlines()
    failures = []
    for category, operation, faithful_shape in TASK3_DIMENSION_ANNOTATIONS:
        matches = [line for line in lines if operation in line]
        if len(matches) != 1:
            failures.append(f"{category}: expected one `{operation}` line, found {len(matches)}")
            continue
        comment = matches[0].partition("#")[2]
        required = ("symbolic", "faithful", "->", faithful_shape)
        missing = [fragment for fragment in required if fragment not in comment]
        if missing:
            failures.append(f"{category}: missing {missing}: {matches[0]}")
    assert not failures, "incomplete Task 3 shape annotations:\n" + "\n".join(failures)


def test_every_task4_ast_transition_has_symbolic_and_faithful_shape_comment():
    failures = []
    found_categories = set()
    for cell_index, source, tree in _task4_implementation_sources():
        comments = _comments_by_line(source)
        sites, categories = _task4_ast_transition_sites(tree)
        found_categories.update(categories)
        lines = source.splitlines()
        for line_number, transition_count in sorted(sites.items()):
            comment = comments.get(line_number, "")
            missing = [
                fragment
                for fragment in ("symbolic", "faithful", "->", "[", "]")
                if fragment not in comment
            ]
            if missing:
                failures.append(
                    f"cell {cell_index}, line {line_number}: missing {missing}: "
                    f"{lines[line_number - 1]}"
                )
            if transition_count > 1:
                failures.append(
                    f"cell {cell_index}, line {line_number}: "
                    f"{transition_count} chained AST transitions: "
                    f"{lines[line_number - 1]}"
                )

    expected_categories = {
        *(f"call:{operation}" for operation in TASK4_AST_SHAPE_CALLS),
        "tensor-index",
        "tensor-iteration",
    }
    assert found_categories == expected_categories
    assert not failures, "incomplete AST-discovered Task 4 annotations:\n" + "\n".join(
        failures
    )


def test_task4_semantic_transitions_have_explicit_shape_comments():
    lines = [
        line
        for _, source, _ in _task4_implementation_sources()
        for line in source.splitlines()
    ]
    failures = []
    for category, operation, expected_count, faithful_shape in (
        TASK4_SEMANTIC_DIMENSION_ANNOTATIONS
    ):
        matches = [line for line in lines if operation in line]
        if len(matches) != expected_count:
            failures.append(
                f"{category}: expected {expected_count} `{operation}` lines, "
                f"found {len(matches)}"
            )
            continue
        for line in matches:
            comment = line.partition("#")[2]
            required = ("symbolic", "faithful", "->", faithful_shape)
            missing = [fragment for fragment in required if fragment not in comment]
            if missing:
                failures.append(f"{category}: missing {missing}: {line}")
    assert not failures, "incomplete semantic Task 4 annotations:\n" + "\n".join(failures)


def test_task4_dimension_changes_have_symbolic_and_faithful_shape_comments():
    lines = code_text(load_notebook()).splitlines()
    failures = []
    for category, operation, faithful_shape in TASK4_DIMENSION_ANNOTATIONS:
        matches = [line for line in lines if operation in line]
        if len(matches) != 1:
            failures.append(f"{category}: expected one `{operation}` line, found {len(matches)}")
            continue
        comment = matches[0].partition("#")[2]
        required = ("symbolic", "faithful", "->", faithful_shape)
        missing = [fragment for fragment in required if fragment not in comment]
        if missing:
            failures.append(f"{category}: missing {missing}: {matches[0]}")
    assert not failures, "incomplete Task 4 shape annotations:\n" + "\n".join(failures)
