"""Time-ordered acceptance tests for multi version-location script
generations.

Every test here builds a real on-disk layout under ``tmp_path`` -- a
script directory with a genuine ``script.py.mako`` and two (or more)
version locations holding real revision files, sometimes with a parent /
child revision pair living in *different* directories -- and drives the
behaviour through the public :class:`.ScriptDirectory` /
:func:`.command.revision` surfaces.

The thing under test is the causal, temporal relationship between:

* the union revision graph (``heads`` / ``walk_revisions``),
* the directory a newly generated revision file actually lands in,
* the generation of discovery that produced the current graph,

as generations are advanced, narrowed and nested.  These are not static
enum/table comparisons: the assertions observe what the *current*
generation sees, what retired generations retain, and what happens when
an inner generation unwinds via an exception.

"""

import os
import shutil
from pathlib import Path

import alembic
import pytest

from alembic import command, util
from alembic.config import Config
from alembic.script import ScriptDirectory, ScriptGeneration
from alembic.testing.fixtures import TestBase

_A = "a" * 12
_B = "b" * 12
_C = "c" * 12
_D = "d" * 12
_E = "e" * 12
_F = "f" * 12


def _revision_body(rid: str, down: str | None) -> str:
    return (
        f'"""revision {rid}"""\n'
        f'revision = "{rid}"\n'
        f"down_revision = {down!r}\n"
        "branch_labels = None\n"
        "depends_on = None\n"
        "def upgrade() -> None:\n    pass\n"
        "def downgrade() -> None:\n    pass\n"
    )


def _write_revision(
    directory: Path,
    rid: str,
    down: str | None,
    *,
    name: str | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name or f"{rid}_rev.py")
    path.write_text(_revision_body(rid, down), encoding="utf-8")
    return path


@pytest.fixture
def world_factory(tmp_path):
    """Build independent on-disk multi-location worlds."""

    template = (
        Path(alembic.__file__).parent
        / "templates"
        / "generic"
        / "script.py.mako"
    )

    def build(name: str):
        root = tmp_path / name
        env = root / "env"
        loc1 = root / "loc1"
        loc2 = root / "loc2"
        for path in (env, loc1, loc2):
            path.mkdir(parents=True)
        shutil.copy(template, env / "script.py.mako")
        return root, env, loc1, loc2

    return build


def _walk_ids(script: ScriptDirectory) -> list[str]:
    return [rev.revision for rev in script.walk_revisions()]


def _make_script_directory(env, *locations, recursive=False):
    return ScriptDirectory(
        env,
        version_locations=list(locations) if locations else None,
        recursive_version_locations=recursive,
    )


class TestUnionGraphAndLanding(TestBase):
    def test_union_walk_sees_both_directories_but_generate_lands_named(
        self, world_factory
    ):
        # parent revision in one location, its child in a *different*
        # location; the union graph must still walk across the boundary.
        root, env, loc1, loc2 = world_factory("union")
        _write_revision(loc1, _A, None, name="a_rev_a.py")
        _write_revision(loc2, _B, _A, name="b_rev_b.py")

        script = _make_script_directory(env, loc1, loc2)

        # heads/walk observe the union
        assert script.get_heads() == [_B]
        assert _walk_ids(script) == [_B, _A]
        assert Path(script.get_revision(_A).path).parent == loc1.absolute()
        assert Path(script.get_revision(_B).path).parent == loc2.absolute()

        # ... while the *new* file lands in the explicitly named
        # location, which diverges from the directory of its own head.
        new = script.generate_revision(
            _C, "rev c", head="head", version_path=str(loc1)
        )
        assert new is not None
        assert new.down_revision == _B
        assert Path(new.path).parent == loc1.absolute()
        assert not (loc2 / f"{_C}_rev_c.py").exists()

        reloaded = _make_script_directory(env, loc1, loc2)
        assert _walk_ids(reloaded) == [_C, _B, _A]
        assert Path(reloaded.get_revision(_C).path).parent == loc1.absolute()

    def test_inferred_landing_diverges_from_explicit_version_path(
        self, world_factory
    ):
        # two independent worlds with the same cross-directory chain
        # a(loc1) -> b(loc2): same head, different landing decision.
        _, env1, l1_1, l1_2 = world_factory("inferred")
        _write_revision(l1_1, _A, None)
        _write_revision(l1_2, _B, _A)
        inferred = _make_script_directory(env1, l1_1, l1_2)

        _, env2, l2_1, l2_2 = world_factory("explicit")
        _write_revision(l2_1, _A, None)
        _write_revision(l2_2, _B, _A)
        explicit = _make_script_directory(env2, l2_1, l2_2)

        inferred_script = inferred.generate_revision(
            _C, "inferred", head="head"
        )
        explicit_script = explicit.generate_revision(
            _C, "explicit", head="head", version_path=str(l2_1)
        )

        # no --version-path: land next to the head script (loc2)...
        assert inferred_script is not None
        assert inferred_script.down_revision == _B
        assert (
            Path(inferred_script.path).parent == l1_2.absolute()
        )
        # ... explicit version_path lands in loc1 even though the head
        # script lives in loc2.
        assert explicit_script is not None
        assert explicit_script.down_revision == _B
        assert (
            Path(explicit_script.path).parent == l2_1.absolute()
        )

        for reloaded in (
            _make_script_directory(env1, l1_1, l1_2),
            _make_script_directory(env2, l2_1, l2_2),
        ):
            assert _walk_ids(reloaded) == [_C, _B, _A]

    def test_generate_rejects_path_outside_current_locations(
        self, world_factory, tmp_path
    ):
        _, env, loc1, loc2 = world_factory("outside")
        _write_revision(loc1, _A, None)
        _write_revision(loc2, _B, _A)
        outside = tmp_path / "elsewhere"
        outside.mkdir()

        script = _make_script_directory(env, loc1, loc2)

        with pytest.raises(
            util.CommandError,
            match="is not represented in current version locations",
        ):
            script.generate_revision(
                _C, "stray", head="head", version_path=str(outside)
            )

        with pytest.raises(
            util.CommandError,
            match="is not represented in current version locations",
        ):
            script.generate_revision(
                _C, "stray", head="head", version_path="foo/bar"
            )

        # nothing was written anywhere
        assert list(loc1.glob("*.py")) == [loc1 / f"{_A}_rev.py"]
        assert list(loc2.glob("*.py")) == [loc2 / f"{_B}_rev.py"]
        assert list(outside.glob("*.py")) == []


class TestSinglePathEntry(TestBase):
    def test_single_path_entry_fails_with_multiple_locations(
        self, world_factory
    ):
        _, env, loc1, loc2 = world_factory("singular")
        _write_revision(loc1, _A, None)
        _write_revision(loc2, _B, _A)
        script = _make_script_directory(env, loc1, loc2)

        with pytest.raises(
            util.CommandError, match="Multiple version_locations present"
        ):
            script.versions
        with pytest.raises(
            util.CommandError, match="Multiple version_locations present"
        ):
            script._singular_version_location

        # narrowing the generation to one location re-enables the
        # single-path entry, and popping restores the failure.
        inner = script.push_generation(version_locations=[loc1])
        try:
            assert Path(script.versions) == loc1.absolute()
            assert script._singular_version_location == loc1.absolute()
        finally:
            script.pop_generation(inner)

        with pytest.raises(
            util.CommandError, match="Multiple version_locations present"
        ):
            script.versions


class TestRecursiveDiscovery(TestBase):
    def test_recursive_flag_changes_discovered_set(self, world_factory):
        _, env, loc1, loc2 = world_factory("recursive")
        _write_revision(loc1, _A, None)
        _write_revision(loc2, _B, _A)
        # this revision only exists in a subdirectory of a location
        _write_revision(
            loc1 / "nested", _D, _A, name="d_nested.py"
        )

        flat = _make_script_directory(env, loc1, loc2, recursive=False)
        assert _walk_ids(flat) == [_B, _A]

        recursive_from_constructor = _make_script_directory(
            env, loc1, loc2, recursive=True
        )
        assert set(_walk_ids(recursive_from_constructor)) == {
            _A,
            _B,
            _D,
        }
        assert set(recursive_from_constructor.get_heads()) == {_B, _D}

        # same ScriptDirectory: a new generation with discovery toggled
        # on produces a different map, popping restores the flat set.
        flat_map = flat.revision_map
        flat.push_generation(recursive_version_locations=True)
        try:
            assert flat.revision_map is not flat_map
            assert set(_walk_ids(flat)) == {_A, _B, _D}
            assert flat.get_revision(_D) is not None
        finally:
            flat.pop_generation()

        assert flat.revision_map is flat_map
        assert _walk_ids(flat) == [_B, _A]
        with pytest.raises(util.CommandError, match="Can't locate revision"):
            flat.get_revision(_D)


class TestGenerationAdvancement(TestBase):
    def test_advanced_generation_retires_old_map_and_locations(
        self, world_factory
    ):
        _, env, loc1, loc2 = world_factory("advance")
        _write_revision(loc1, _A, None)
        _write_revision(loc2, _B, _A)
        script = _make_script_directory(env, loc1, loc2)

        # freeze generation 0's union graph
        assert _walk_ids(script) == [_B, _A]
        map0 = script.revision_map
        assert script.current_generation.generation_id == 0

        inner: ScriptGeneration = script.push_generation(
            version_locations=[loc1]
        )
        try:
            assert inner.generation_id == 1
            # a fresh map, not a memo reset on the old one
            assert script.revision_map is not map0
            assert script._version_locations == [loc1.absolute()]

            # the revision living in the removed location is a ghost no
            # longer, not a stale memo entry
            assert _walk_ids(script) == [_A]
            with pytest.raises(
                util.CommandError, match="Can't locate revision"
            ):
                script.get_revision(_B)

            # the removed location can no longer be a landing target
            with pytest.raises(
                util.CommandError,
                match="is not represented in current version locations",
            ):
                script.generate_revision(
                    _C, "stray", head=_A, version_path=str(loc2)
                )
            assert not list(loc2.glob(f"{_C}*"))

            # generating inside the new generation lands in, and joins,
            # the new graph only
            new_in_inner = script.generate_revision(
                _E, "inner", head=_A, version_path=str(loc1)
            )
            assert new_in_inner is not None
            assert Path(new_in_inner.path).parent == loc1.absolute()
            assert _walk_ids(script) == [_E, _A]
        finally:
            script.pop_generation(inner)

        # generation 0 is restored exactly: same map object, frozen
        # content, multi-location rules back
        assert script.current_generation.generation_id == 0
        assert script.revision_map is map0
        assert _walk_ids(script) == [_B, _A]
        assert script.get_revision(_B) is not None
        with pytest.raises(util.CommandError, match="Can't locate revision"):
            script.get_revision(_E)
        assert script._version_locations == [
            loc1.absolute(),
            loc2.absolute(),
        ]

        # advancing again builds a fresh discovery generation that sees
        # what was meanwhile written to disk, while the retired map0
        # stays frozen; the two ledgers never consult each other.
        second = script.push_generation(version_locations=[loc1])
        try:
            assert script.revision_map is not map0
            assert set(_walk_ids(script)) == {_A, _E}
            with pytest.raises(
                util.CommandError, match="Can't locate revision"
            ):
                script.get_revision(_B)
        finally:
            script.pop_generation(second)

        # retired generation still answers from its frozen snapshot and
        # remains usable when it is current again; loc2 is a legal
        # landing target again
        restored = script.generate_revision(
            _F, "outer", head=_B, version_path=str(loc2)
        )
        assert restored is not None
        assert Path(restored.path).parent == loc2.absolute()
        assert script.revision_map is map0
        assert _walk_ids(script) == [_F, _B, _A]


class TestNestedGeneration(TestBase):
    def test_inner_generation_isolated_and_unwinds_on_exception(
        self, world_factory
    ):
        _, env, loc1, loc2 = world_factory("nested")
        _write_revision(loc1, _A, None)
        _write_revision(loc2, _B, _A)
        script = _make_script_directory(env, loc1, loc2)

        assert _walk_ids(script) == [_B, _A]
        outer = script.current_generation
        outer_map = script.revision_map

        with pytest.raises(RuntimeError, match="boom"):
            with script.generation(
                version_locations=[loc1], version_path=loc1
            ) as inner:
                assert inner is not outer
                # inside, the union is narrowed and landing is pinned;
                # no --version-path is needed even with locations set
                new = script.generate_revision(_E, "inner", head="base")
                assert new is not None
                assert Path(new.path).parent == loc1.absolute()
                assert set(_walk_ids(script)) == {_A, _E}
                with pytest.raises(
                    util.CommandError, match="Can't locate revision"
                ):
                    script.get_revision(_B)
                raise RuntimeError("boom")

        # exceptional unwind restores everything
        assert len(script._generation_stack) == 1
        assert script.current_generation is outer
        assert script.revision_map is outer_map
        assert _walk_ids(script) == [_B, _A]
        with pytest.raises(util.CommandError, match="Can't locate revision"):
            script.get_revision(_E)
        assert script._version_locations == [
            loc1.absolute(),
            loc2.absolute(),
        ]
        with pytest.raises(
            util.CommandError, match="Multiple version_locations present"
        ):
            script.versions

    def test_inner_landing_intent_does_not_leak_to_outer(
        self, world_factory
    ):
        _, env, loc1, loc2 = world_factory("intent")
        script = _make_script_directory(env, loc1, loc2)

        # freeze an empty outer union graph
        assert script.get_heads() == []
        outer_map = script.revision_map

        with pytest.raises(
            util.CommandError, match="please specify --version-path"
        ):
            script.generate_revision(_C, "outer", head="base")

        with script.generation(version_locations=script.version_locations):
            pass  # context manager with inherited locations works

        with script.generation(version_path=loc2) as inner:
            # same location list as the outer generation, but the inner
            # one pins where unnamed revisions land
            assert script._version_locations == [
                loc1.absolute(),
                loc2.absolute(),
            ]
            new = script.generate_revision(_E, "inner", head="base")
            assert new is not None
            assert Path(new.path).parent == loc2.absolute()
            assert script.get_heads() == [_E]
            assert inner.default_version_path == loc2.absolute()

        # the file physically exists in a shared directory, but the
        # outer generation's frozen map never scanned it in
        assert script.revision_map is outer_map
        assert script.get_heads() == []
        with pytest.raises(util.CommandError, match="Can't locate revision"):
            script.get_revision(_E)
        assert script.current_generation.default_version_path is None
        with pytest.raises(
            util.CommandError, match="please specify --version-path"
        ):
            script.generate_revision(_F, "outer again", head="base")

    def test_pop_generation_rules(self, world_factory):
        _, env, loc1, loc2 = world_factory("poprules")
        script = _make_script_directory(env, loc1, loc2)

        with pytest.raises(
            util.CommandError, match="Cannot pop the initial script generation"
        ):
            script.pop_generation()

        first = script.push_generation()
        second = script.push_generation()
        with pytest.raises(
            util.CommandError, match="nesting mismatch"
        ):
            script.pop_generation(first)

        # mismatched pop changed nothing
        assert script.current_generation is second
        script.pop_generation(second)
        assert script.current_generation is first
        script.pop_generation()
        assert len(script._generation_stack) == 1


class TestCommandRevisionWiring(TestBase):
    def _config(self, env, *locations):
        cfg = Config()
        cfg.set_main_option("script_location", str(env))
        cfg.set_main_option(
            "version_locations", os.pathsep.join(str(p) for p in locations)
        )
        cfg.set_main_option("path_separator", "os")
        return cfg

    def test_command_revision_narrows_generation_and_restores(
        self, world_factory
    ):
        _, env, loc1, loc2 = world_factory("command")
        _write_revision(loc1, _A, None)
        _write_revision(loc2, _B, _A)

        cfg = self._config(env, loc1, loc2)

        # unchanged pin behaviour: multi locations + no version path is
        # an error through command.revision as well
        with pytest.raises(
            util.CommandError, match="please specify --version-path"
        ):
            command.revision(cfg, message="multi", head="base")

        # the host-side override enters a narrowed nested generation for
        # the duration of the command; no version_path is needed with a
        # single location, and the file really lands in that location
        script = command.revision(
            cfg,
            message="narrow",
            head="base",
            rev_id=_C,
            version_locations=[str(loc1)],
        )
        assert script is not None
        assert Path(script.path).parent == loc1.absolute()
        assert (loc1 / f"{_C}_narrow.py").exists()
        assert not list(loc2.glob(f"{_C}*"))

        # a fresh script directory built from the unchanged config still
        # sees the configured union
        reloaded = ScriptDirectory.from_config(cfg)
        assert set(_walk_ids(reloaded)) == {_A, _B, _C}

        # out-of-bounds version_path under the override is rejected
        with pytest.raises(
            util.CommandError,
            match="is not represented in current version locations",
        ):
            command.revision(
                cfg,
                message="stray",
                head="base",
                rev_id=_D,
                version_locations=[str(loc1)],
                version_path=str(loc2),
            )
        assert not list(loc2.glob(f"{_D}*"))

    def test_command_revision_recursive_override(self, world_factory):
        _, env, loc1, loc2 = world_factory("command_recursive")
        _write_revision(loc1, _A, None)
        _write_revision(loc1 / "nested", _D, _A, name="d_nested.py")
        _write_revision(loc2, _B, _A)

        cfg = self._config(env, loc1, loc2)

        # before the override, flat discovery cannot see the nested
        # revision, so it cannot be used as a head through the command
        with pytest.raises(
            util.CommandError, match="Can't locate revision"
        ):
            command.revision(
                cfg, message="flat", head=_D, rev_id=_F
            )
        assert not list(loc1.glob(f"{_F}*"))

        # the override only changes discovery for this command; the
        # nested revision is a selectable head while it is in effect
        script = command.revision(
            cfg,
            message="recursive",
            head=_D,
            rev_id=_E,
            version_path=str(loc1),
            recursive_version_locations=True,
        )
        assert script is not None
        assert Path(script.path).parent == loc1.absolute()
        assert script.down_revision == _D
        assert (loc1 / f"{_E}_recursive.py").exists()
