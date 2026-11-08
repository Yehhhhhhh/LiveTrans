"""backend/tools/freeze_lock.py: the dependency closure written to requirements.lock."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_spec = importlib.util.spec_from_file_location(
    "freeze_lock", Path(__file__).resolve().parent.parent / "tools" / "freeze_lock.py"
)
freeze_lock = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(freeze_lock)
Requirement = freeze_lock.Requirement


def fake_distributions(monkeypatch, packages):
    """packages: name -> list of requirement strings (its Requires-Dist)."""
    dists = {
        name: SimpleNamespace(version="1.0", requires=reqs, metadata={"Name": name})
        for name, reqs in packages.items()
    }

    def distribution(name):
        try:
            return dists[freeze_lock.canonicalize_name(name)]
        except KeyError:
            raise freeze_lock.metadata.PackageNotFoundError(name) from None

    monkeypatch.setattr(freeze_lock.metadata, "distribution", distribution)


def test_cycle_through_extras_terminates(monkeypatch):
    # a[x] needs b[y], and b[y] needs a[x] again.
    fake_distributions(monkeypatch, {
        "a": ["b[y]; extra == 'x'", "c"],
        "b": ["a[x]; extra == 'y'"],
        "c": [],
    })
    assert sorted(freeze_lock.closure([Requirement("a[x]")])) == ["a", "b", "c"]


def test_extras_only_pull_in_their_own_dependencies(monkeypatch):
    fake_distributions(monkeypatch, {
        "app": ["web[server]", "lib"],
        "web": ["uvicorn; extra == 'server'", "pytest; extra == 'test'"],
        "lib": [],
        "uvicorn": [],
        "pytest": [],
    })
    assert sorted(freeze_lock.closure([Requirement("app")])) == ["app", "lib", "uvicorn", "web"]


def test_package_seen_first_without_extras_still_gets_them_later(monkeypatch):
    fake_distributions(monkeypatch, {
        "root1": ["web"],
        "root2": ["web[server]"],
        "web": ["uvicorn; extra == 'server'"],
        "uvicorn": [],
    })
    roots = [Requirement("root2"), Requirement("root1")]  # todo is a stack: root1 is expanded first
    assert "uvicorn" in freeze_lock.closure(roots)


def test_platform_markers_are_respected(monkeypatch):
    fake_distributions(monkeypatch, {"app": ["winonly; sys_platform == 'nonexistent-os'"]})
    assert sorted(freeze_lock.closure([Requirement("app")])) == ["app"]


def test_missing_package_stops_with_a_message(monkeypatch):
    fake_distributions(monkeypatch, {"app": ["absent"]})
    with pytest.raises(SystemExit) as e:
        freeze_lock.closure([Requirement("app")])
    assert "absent is required but not installed" in str(e.value)
