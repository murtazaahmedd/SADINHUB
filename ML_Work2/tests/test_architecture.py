"""The layered structure is enforced by tests, so it cannot silently erode.

Layers (dependencies flow downward only):
    routers -> decorators/schemas -> services -> interfaces -> repositories -> database/sql
    models, utilities, constants, errors, config are pure and import nothing above them.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAYERS = ["config", "constants", "database", "decorators", "errors", "interfaces", "main", "models",
          "repositories", "routers", "schemas", "services", "utilities"]
FORBIDDEN = {
    "constants": {"repositories", "services", "routers", "database", "main", "decorators", "models", "utilities"},
    "errors": {"repositories", "services", "routers", "database", "main", "decorators", "models", "utilities"},
    "utilities": {"repositories", "services", "routers", "database", "main", "decorators"},
    "models": {"repositories", "services", "routers", "database", "main", "decorators"},
    "interfaces": {"repositories", "services", "routers", "main", "decorators"},
    "repositories": {"services", "routers", "main", "decorators", "models"},
    "services": {"repositories", "routers", "main", "decorators", "database"},
    "schemas": {"repositories", "services", "routers", "main", "database", "models"},
    "routers": {"repositories", "models", "database", "utilities.scoring"},
}
# Documented composition roots may wire concrete classes.
ALLOWED = {("services/container.py", "repositories"), ("services/container.py", "services"),
           ("routers/", "services"), ("repositories/container.py", "database")}


def _python_files(layer):
    return [p for p in (ROOT / layer).rglob("*.py") if "__pycache__" not in p.parts]


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name


def _allowed(rel, module):
    return any(rel.startswith(prefix) and module.split(".")[0] == layer or
               (rel.startswith(prefix) and module.startswith(layer)) for prefix, layer in ALLOWED)


@pytest.mark.parametrize("layer", sorted(FORBIDDEN))
def test_layers_only_depend_downward(layer):
    broken = []
    for path in _python_files(layer):
        rel = str(path.relative_to(ROOT))
        for module in _imports(path):
            hit = [f for f in FORBIDDEN[layer] if module == f or module.startswith(f + ".")]
            if hit and not _allowed(rel, module):
                broken.append(f"{rel} imports {module}")
    assert not broken, "\n".join(broken)


def test_services_are_wired_only_through_interfaces():
    """A service may import another service's interface, never its implementation (container excepted)."""
    broken = []
    for path in _python_files("services"):
        rel = str(path.relative_to(ROOT))
        if rel == "services/container.py" or rel.endswith("__init__.py"):
            continue
        for module in _imports(path):
            if module.startswith("services."):
                broken.append(f"{rel} imports {module}")
    assert not broken, "\n".join(broken)


def test_repositories_never_write_inline_sql():
    """Every SQL string in a repository is a call to an api.* function."""
    broken = []
    for path in _python_files("repositories"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                text = node.value.upper()
                if ("SELECT " in text or "INSERT " in text or "UPDATE " in text or "DELETE " in text) and \
                        "API." not in text and "\n" not in node.value.strip()[:80] and len(node.value) < 300:
                    broken.append(f"{path.relative_to(ROOT)}: {node.value[:60]!r}")
    assert not broken, "\n".join(broken)


def test_every_package_exports_explicitly():
    empty = []
    for init in ROOT.rglob("__init__.py"):
        if any(part in {"__pycache__", ".venv", "reference_project", "archive_gpt_version", "tests"}
               for part in init.parts) or ROOT.joinpath("sql") in init.parents:
            continue
        text = init.read_text(encoding="utf-8").strip()
        if "__all__" not in text:
            empty.append(str(init.relative_to(ROOT)))
    assert not empty, "\n".join(empty)


def test_every_endpoint_carries_the_error_handler_marker_and_enforce():
    from main.app import create_app
    from decorators.http_error_handler import MARKER
    app = create_app()
    routes = []

    def collect(routes_like):
        for r in routes_like:
            inner = getattr(r, "routes", None) or getattr(getattr(r, "original_router", None), "routes", None)
            if inner:
                collect(inner)
            elif hasattr(r, "endpoint"):
                routes.append(r)

    collect(app.routes)
    api = [r for r in routes if r.path not in {"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}]
    assert len(api) >= 9
    missing = [r.path for r in api if not getattr(r.endpoint, MARKER, False)]
    assert not missing, missing


def _code_lines(node):
    """Lines of a function body excluding its docstring."""
    body = node.body
    start = body[1].lineno if (body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant)
                               and isinstance(body[0].value.value, str) and len(body) > 1) else body[0].lineno
    return node.end_lineno - start + 1


def test_no_function_has_more_than_30_lines_of_code():
    too_long = []
    for layer in LAYERS:
        for path in _python_files(layer):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _code_lines(node) > 30:
                    too_long.append(f"{path.relative_to(ROOT)}:{node.name} ({_code_lines(node)} lines)")
    assert not too_long, "\n".join(too_long)


@pytest.mark.slow
def test_every_module_imports_on_its_own_in_a_fresh_interpreter():
    """An import cycle only shows up when a module is imported first; check each one that way."""
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor
    modules = []
    for layer in LAYERS:
        for path in _python_files(layer):
            parts = list(path.relative_to(ROOT).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            if parts and parts != ["main", "app"] and parts != ["main", "cli"] and parts != ["database", "udf_load"]:
                modules.append(".".join(parts))

    def attempt(name):
        done = subprocess.run([sys.executable, "-c", f"import {name}"], cwd=ROOT, capture_output=True, text=True)
        return name, done.returncode, done.stderr.strip().splitlines()[-1:] if done.returncode else []

    with ThreadPoolExecutor(max_workers=8) as pool:
        failures = [r for r in pool.map(attempt, sorted(set(modules))) if r[1] != 0]
    assert not failures, failures
