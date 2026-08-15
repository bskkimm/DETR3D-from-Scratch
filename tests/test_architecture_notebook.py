import ast
import json
import tokenize
from io import StringIO
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
IMAGE_FEATURE_MARKERS = {"ARCH-GRID-MASK", "ARCH-BACKBONE", "ARCH-FPN"}
IMAGE_FEATURE_DEFINITIONS = {
    "GridMask",
    "ModulatedDeformConv2dPack",
    "MultiViewImageBackbone",
    "ImageFPN",
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
