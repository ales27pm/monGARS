import importlib
import importlib.machinery
import importlib.util
import sys
import types

import pytest

MODULE_NAME = "monGARS.mlops._unsloth_bootstrap"


def _reload_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    *,
    spec: importlib.machinery.ModuleSpec | None,
    module: types.ModuleType | None,
    zoo_spec: importlib.machinery.ModuleSpec | None = None,
    import_error: bool = False,
):
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args: object, **kwargs: object):
        if name == "unsloth":
            return spec
        if name == "unsloth_zoo":
            return zoo_spec
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    original_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "unsloth":
            if import_error:
                raise ImportError("boom")
            if module is not None:
                sys.modules.setdefault("unsloth", module)
                return module
        if name == "unsloth_zoo" and zoo_spec is not None:
            zoo_module = sys.modules.setdefault(
                "unsloth_zoo", types.ModuleType("unsloth_zoo")
            )
            return zoo_module
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    if MODULE_NAME in sys.modules:
        monkeypatch.setitem(sys.modules, MODULE_NAME, sys.modules[MODULE_NAME])
    monkeypatch.delitem(sys.modules, MODULE_NAME, raising=False)

    if module is None:
        monkeypatch.delitem(sys.modules, "unsloth", raising=False)
    else:
        monkeypatch.setitem(sys.modules, "unsloth", module)

    if zoo_spec is None:
        monkeypatch.delitem(sys.modules, "unsloth_zoo", raising=False)
    else:
        monkeypatch.setitem(sys.modules, "unsloth_zoo", types.ModuleType("unsloth_zoo"))

    module_obj = importlib.import_module(MODULE_NAME)
    return importlib.reload(module_obj)


def test_bootstrap_detects_missing_unsloth(monkeypatch: pytest.MonkeyPatch) -> None:
    reloaded = _reload_bootstrap(monkeypatch, spec=None, module=None)
    assert reloaded.UNSLOTH_AVAILABLE is False


def test_bootstrap_imports_unsloth_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_spec = importlib.machinery.ModuleSpec("unsloth", loader=None)
    fake_module = types.ModuleType("unsloth")
    reloaded = _reload_bootstrap(
        monkeypatch,
        spec=fake_spec,
        module=fake_module,
        zoo_spec=importlib.machinery.ModuleSpec("unsloth_zoo", loader=None),
    )
    assert reloaded.UNSLOTH_AVAILABLE is True
    assert sys.modules.get("unsloth") is fake_module


def test_bootstrap_tolerates_missing_unsloth_zoo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_spec = importlib.machinery.ModuleSpec("unsloth", loader=None)
    fake_module = types.ModuleType("unsloth")
    reloaded = _reload_bootstrap(
        monkeypatch,
        spec=fake_spec,
        module=fake_module,
        zoo_spec=None,
    )
    assert reloaded.UNSLOTH_AVAILABLE is True
    assert sys.modules.get("unsloth") is fake_module


def test_bootstrap_handles_import_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_spec = importlib.machinery.ModuleSpec("unsloth", loader=None)
    fake_module = types.ModuleType("unsloth")
    reloaded = _reload_bootstrap(
        monkeypatch,
        spec=fake_spec,
        module=fake_module,
        import_error=True,
    )
    assert reloaded.UNSLOTH_AVAILABLE is False
    assert "unsloth" not in sys.modules
