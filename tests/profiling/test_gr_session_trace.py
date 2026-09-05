# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only synthetic Nsight schema; no Torch, CUDA, or model imports."""

import sqlite3

import pytest

from tests.profiling.gr_session_trace import analyze_trace


def _database(tmp_path):
    path = tmp_path / "trace.sqlite"
    db = sqlite3.connect(path)
    db.executescript("""
        create table StringIds(id integer, value text);
        create table NVTX_EVENTS(start integer, end integer, globalTid integer, text text, textId integer);
        create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer, end integer, globalTid integer,
            correlationId integer, nameId integer, returnValue integer);
        create table CUPTI_ACTIVITY_KIND_MEMCPY(start integer, end integer, globalPid integer,
            correlationId integer, bytes integer, copyKind integer);
    """)
    db.executemany("insert into StringIds values (?,?)", [
        (1, "cudaEventSynchronize"), (2, "cudaEventQuery"), (3, "cudaMemcpyAsync")])
    tid = (17 << 24) + 1
    def mark(start, end, name):
        db.execute("insert into NVTX_EVENTS values (?,?,?,?,null)", (start, end, tid, name))
    for start, end, name in [
        (10, 15, "gr_async_schedule"), (20, 25, "gr_prefill_bootstrap"),
        (26, 27, "gr_progress_output"), (30, 35, "gr_async_schedule"),
        (40, 41, "gr_progress_output"),
        (42, 49, "x.EngineCore.step_with_batch_queue"),
        (43, 44, "x.AsyncOutputFuture.result"), (45, 48, "x.GRAsyncScheduler.update_from_output"),
        (50, 55, "gr_async_schedule"), (60, 70, "gr_final_output"),
        (71, 79, "x.EngineCore.step_with_batch_queue"),
        (72, 73, "x.AsyncOutputFuture.result"), (74, 78, "x.GRAsyncScheduler.update_from_output"),
        (80, 99, "x.EngineCore.step_with_batch_queue"),
        (81, 82, "gr_async_schedule"),  # Empty housekeeping dispatch, final FIFO drain.
        (86, 90, "x.AsyncOutputFuture.result"), (91, 98, "x.GRAsyncScheduler.update_from_output"),
        (101, 102, "x.GRAsyncScheduler.update_from_output"),
    ]:
        mark(start, end, name)
    db.executemany("insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?,?,?,?,?,0)", [
        (36, 37, tid, 1, 2), (62, 63, tid, 2, 3), (75, 76, tid, 3, 3),
        (87, 89, tid, 4, 1), (103, 104, tid, 5, 1)])
    db.executemany("insert into CUPTI_ACTIVITY_KIND_MEMCPY values (?,?,?,?,?,2)", [
        (64, 65, 17 << 24, 2, 1024), (83, 84, 17 << 24, 3, 8),
        (83, 84, 18 << 24, 3, 99999)])  # Other PID must not contaminate count.
    db.commit()
    db.close()
    return path


def _modify(path, sql):
    with sqlite3.connect(path) as db:
        db.executescript(sql)


def test_window_includes_final_tail_copy_and_future_wait(tmp_path):
    result = analyze_trace(_database(tmp_path), 3)["sessions"][0]
    assert (result["start_ns"], result["end_ns"]) == (10, 98)
    assert result["reconcile_count"] == 3
    assert result["housekeeping_reconciles_after_session"] == 1
    assert (result["d2h_count"], result["d2h_bytes"]) == (2, 1032)
    assert [r["final_relation"] for r in result["d2h"]] == ["inside_final", "after_final"]
    sync = result["synchronize_query_apis"][-1]
    assert sync["name"] == "cudaEventSynchronize"
    assert sync["final_relation"] == "after_final"
    assert any(r["name"].endswith("AsyncOutputFuture.result") for r in sync["nvtx_ranges"])
    assert result["api_counts"] == {"cudaEventQuery": 1, "cudaEventSynchronize": 1}
    assert result["omitted_by_final_end_start_filter"] == {
        "d2h_count": 1, "d2h_bytes": 8, "api_counts": {"cudaEventSynchronize": 1}}


@pytest.mark.parametrize("sql", [
    "delete from NVTX_EVENTS where start=91",  # Final reconcile missing, housekeeping is not a substitute.
    "delete from NVTX_EVENTS where start=86",
    "update NVTX_EVENTS set end=null where start=91",
    "delete from NVTX_EVENTS where start=60",
    "delete from NVTX_EVENTS where start=20",
    "delete from NVTX_EVENTS where start=80",
    "insert into NVTX_EVENTS select * from NVTX_EVENTS where start=80",
    "insert into NVTX_EVENTS values(103,104,285212673,'x.AsyncOutputFuture.result',null)",
    "update NVTX_EVENTS set globalTid=globalTid+1 where start=91",
    "update NVTX_EVENTS set start=23 where start=26",  # Output before bootstrap ends.
    "insert into NVTX_EVENTS values(105,106,285212673,'gr_progress_output',null)",
])
def test_incomplete_or_ambiguous_trace_rejected(tmp_path, sql):
    path = _database(tmp_path)
    _modify(path, sql)
    with pytest.raises(ValueError):
        analyze_trace(path, 3)


@pytest.mark.parametrize("stages", [0, -1, True, 2, 4])
def test_explicit_generation_count_enforced(tmp_path, stages):
    with pytest.raises(ValueError):
        analyze_trace(_database(tmp_path), stages)


def test_copy_crossing_final_boundary_not_misclassified(tmp_path):
    path = _database(tmp_path)
    _modify(path, "update CUPTI_ACTIVITY_KIND_MEMCPY set start=69,end=72 where bytes=1024")
    copy = analyze_trace(path, 3)["sessions"][0]["d2h"][0]
    assert copy["final_relation"] == "overlaps_final_boundary"


def test_missing_capture_table_rejected(tmp_path):
    path = _database(tmp_path)
    _modify(path, "drop table CUPTI_ACTIVITY_KIND_RUNTIME")
    with pytest.raises(ValueError, match="missing CUDA API"):
        analyze_trace(path, 3)


def test_independent_pids_can_have_same_timestamps(tmp_path):
    path = _database(tmp_path)
    _modify(path, """
        insert into NVTX_EVENTS select start,end,globalTid+16777216,text,textId from NVTX_EVENTS;
        insert into CUPTI_ACTIVITY_KIND_RUNTIME
            select start,end,globalTid+16777216,correlationId,nameId,returnValue from CUPTI_ACTIVITY_KIND_RUNTIME;
    """)
    sessions = analyze_trace(path, 3)["sessions"]
    assert [s["pid"] for s in sessions] == [17, 18]
    assert [s["d2h_bytes"] for s in sessions] == [1032, 99999]


def test_two_serial_sessions_do_not_share_reconcile_or_tail(tmp_path):
    path = _database(tmp_path)
    _modify(path, """
        insert into NVTX_EVENTS select start+200,end+200,globalTid,text,textId from NVTX_EVENTS;
        insert into CUPTI_ACTIVITY_KIND_RUNTIME
            select start+200,end+200,globalTid,correlationId,nameId,returnValue from CUPTI_ACTIVITY_KIND_RUNTIME;
        insert into CUPTI_ACTIVITY_KIND_MEMCPY
            select start+200,end+200,globalPid,correlationId,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY;
    """)
    sessions = analyze_trace(path, 3)["sessions"]
    assert [(s["start_ns"], s["end_ns"]) for s in sessions] == [(10, 98), (210, 298)]
    assert [s["d2h_bytes"] for s in sessions] == [1032, 1032]


def test_single_stage_parent_can_start_before_prefill_schedule(tmp_path):
    path = _database(tmp_path)
    _modify(path, """
        delete from NVTX_EVENTS where start not in (10,20,26,42,43,45);
        update NVTX_EVENTS set text='gr_final_output' where start=26;
        update NVTX_EVENTS set start=5 where start=42;
    """)
    session = analyze_trace(path, 1)["sessions"][0]
    assert (session["start_ns"], session["end_ns"], session["reconcile_count"]) == (10, 48, 1)


def test_copy_crossing_session_start_is_included_and_flagged(tmp_path):
    path = _database(tmp_path)
    _modify(path, "update CUPTI_ACTIVITY_KIND_MEMCPY set start=9,end=11 where bytes=1024")
    copy = analyze_trace(path, 3)["sessions"][0]["d2h"][0]
    assert copy["crosses_session_boundary"] is True
    assert copy["final_relation"] == "before_final"
