"""Sandboxes and the minielt loader."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import duckdb
import pytest
import yaml

from elt_rl.minielt import cli as minielt
from elt_rl.sandbox.docker import docker_run_argv
from elt_rl.sandbox.local import LocalProcessSandbox, truncate_middle


def test_truncate_middle_keeps_head_and_tail() -> None:
    assert truncate_middle("short", 100) == "short"
    text = "HEAD" + "x" * 1000 + "TAIL"
    cut = truncate_middle(text, 60)
    assert cut.startswith("HEAD") and cut.endswith("TAIL") and "characters omitted" in cut
    assert len(cut) < 120


async def test_local_sandbox_runs_in_workspace_and_kills_on_timeout(tmp_path: Path) -> None:
    sandbox = LocalProcessSandbox(tmp_path / "ws")
    result = await sandbox.run_command("pwd && echo $HOME && echo err >&2; exit 3")
    assert result.exit_code == 3 and result.stderr.strip() == "err"
    assert result.stdout.split() == [str(tmp_path / "ws"), str(tmp_path / "ws" / ".home")]

    await sandbox.write_file("elt/a/b.txt", "data")
    assert (await sandbox.read_file("elt/a/b.txt")).stdout == "data"
    assert (await sandbox.read_file("missing.txt")).exit_code == 1

    # The whole process tree dies with the command, not just the shell.
    started = time.monotonic()
    result = await sandbox.run_command("(sleep 60; touch late.txt) & sleep 60", timeout=1)
    assert result.exit_code == 124 and time.monotonic() - started < 10
    assert "timed out" in result.stderr

    # A detached background job survives the command and is killed on request.
    await sandbox.run_command("nohup sh -c 'sleep 2; touch late.txt' > /dev/null 2>&1 &")
    await sandbox.kill_background()
    await asyncio.sleep(2.5)
    assert not (tmp_path / "ws" / "late.txt").exists()
    await sandbox.cleanup()


def test_docker_command_line_isolates_the_rollout(tmp_path: Path) -> None:
    argv = docker_run_argv(
        "elt-rl-x", "elt-swe", tmp_path, "elt-docker_elt_network", ("elt",), uid=1000, gid=1000
    )
    joined = " ".join(argv)
    mount = str(tmp_path.resolve())
    assert f"-v {mount}:/workspace:ro" in joined  # task files are read-only
    assert f"-v {mount}/elt:/workspace/elt" in joined  # except the working directory
    assert "--network elt-docker_elt_network" in joined
    assert "--cap-drop ALL" in joined and "no-new-privileges" in joined
    assert "--user 1000:1000" in joined and argv[-3:] == ["elt-swe", "sleep", "infinity"]
    # The hermetic local benchmark runs with no network at all.
    local = docker_run_argv("n", "elt-rl-local", tmp_path, None, ("elt", "warehouse"))
    assert local[local.index("--network") + 1] == "none"


# -- minielt --------------------------------------------------------------------------------
@pytest.fixture
def pipeline(tmp_path: Path) -> dict:
    (tmp_path / "bucket").mkdir()
    (tmp_path / "bucket" / "city.csv").write_text("id,name\n1,Oslo\n2,Lima\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "event.jsonl").write_text('{"id": 1, "kind": "a"}\n{"id": 2, "kind": "b"}\n')
    return {
        "destination": {"type": "duckdb", "path": str(tmp_path / "w.duckdb"), "schema": "s"},
        "sources": [
            {"name": "bucket", "type": "object_store", "path": str(tmp_path / "bucket")},
            {"name": "docs", "type": "documents", "path": "docs"},  # relative to the cwd
        ],
        "connections": [
            {"name": "bucket_to_wh", "source": "bucket",
             "streams": [{"name": "city", "sync_mode": "full_refresh_append"}]},
            {"name": "docs_to_wh", "source": "docs",
             "streams": [{"name": "event", "sync_mode": "full_refresh_overwrite"}]},
        ],
    }  # fmt: skip


def run_minielt(tmp_path: Path, *argv: str) -> int:
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        return minielt.main(list(argv))
    finally:
        os.chdir(cwd)


def test_minielt_apply_sync_status(tmp_path: Path, pipeline: dict, capsys) -> None:
    (tmp_path / "elt").mkdir()
    (tmp_path / "elt" / "pipeline.yaml").write_text(yaml.safe_dump(pipeline))
    assert run_minielt(tmp_path, "apply", "-f", "elt/pipeline.yaml") == 0
    assert run_minielt(tmp_path, "sync", "--all") == 0
    assert run_minielt(tmp_path, "status") == 0
    assert "succeeded" in capsys.readouterr().out

    def counts() -> dict[str, int]:
        con = duckdb.connect(str(tmp_path / "w.duckdb"), read_only=True)
        try:
            return {t: con.execute(f"select count(*) from s.{t}").fetchall()[0][0] for t in ("city", "event")}
        finally:
            con.close()

    assert counts() == {"city": 2, "event": 2}
    con = duckdb.connect(str(tmp_path / "w.duckdb"), read_only=True)
    columns = [r[0] for r in con.execute("describe s.city").fetchall()]
    con.close()
    assert {"id", "name", "_airbyte_raw_id", "_airbyte_extracted_at"} <= set(columns)

    # Like Airbyte: an append stream synced again is loaded again; overwrite is not.
    assert run_minielt(tmp_path, "sync", "--all") == 0
    assert counts() == {"city": 4, "event": 2}


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda p: p["destination"].update(type="postgres"), "destination.type"),
        (lambda p: p["sources"][0].update(type="ftp"), "unknown type"),
        (lambda p: p["connections"][0].update(source="nope"), "nope"),
        (lambda p: p["connections"][0]["streams"][0].update(sync_mode="incremental"), "sync_mode"),
        (lambda p: p["connections"][0]["streams"][0].update(name="ghost"), "ghost"),
        (lambda p: p.pop("sources"), "sources"),
    ],
)
def test_minielt_rejects_bad_pipelines_with_a_useful_message(
    tmp_path: Path, pipeline: dict, mutate, message: str, capsys
) -> None:
    mutate(pipeline)
    (tmp_path / "elt").mkdir()
    (tmp_path / "elt" / "pipeline.yaml").write_text(yaml.safe_dump(pipeline))
    code = run_minielt(tmp_path, "apply", "-f", "elt/pipeline.yaml")
    captured = capsys.readouterr()
    if code == 0:  # whether a stream exists is only known when its job runs
        code = run_minielt(tmp_path, "sync", "--all")
        captured = capsys.readouterr()
    assert code != 0
    assert message in captured.out + captured.err
