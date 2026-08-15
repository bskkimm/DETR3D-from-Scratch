import ast
import json
import tokenize
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from detr3d.models.transformer.cross_attention import (
    Detr3DCrossAttention as ProductionDetr3DCrossAttention,
)
from detr3d.models.transformer.feature_sampling import (
    feature_sampling as production_feature_sampling,
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
APPROVED_PRODUCTION_IMPORTS = {
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


def _forbidden_imports(tree):
    forbidden = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_name = alias.name.rsplit(".", 1)[-1]
                is_model_module = alias.name == "detr3d.models" or alias.name.startswith(
                    "detr3d.models."
                )
                if (
                    is_model_module
                    or imported_name in DEFINITIONS
                    or alias.asname in DEFINITIONS
                ):
                    forbidden.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            is_model_module = module == "detr3d.models" or module.startswith(
                "detr3d.models."
            )
            for alias in node.names:
                imported_symbol = f"{module}.{alias.name}"
                is_approved = (module, alias.name) in APPROVED_PRODUCTION_IMPORTS
                if (
                    alias.name == "*"
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
