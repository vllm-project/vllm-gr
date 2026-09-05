# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only, conservative session windows for serial UniProc GR Nsight traces.

The window ends at the final output's AsyncOutputFuture reconciliation, NOT
at gr_final_output.end. Markers have no request IDs: this reader supports only
the validated, single-thread, serial GR topology below, not arbitrary mixed or
multiprocess workloads. API duration is Host time, not device stall time.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3


SCHEDULE = "gr_async_schedule"
BOOTSTRAP = "gr_prefill_bootstrap"
PROGRESS = "gr_progress_output"
FINAL = "gr_final_output"
RECONCILE = "GRAsyncScheduler.update_from_output"
FUTURE = "AsyncOutputFuture.result"
ENGINE = "EngineCore.step_with_batch_queue"


def _is(row, name):
    return row["name"] == name or row["name"].endswith("." + name)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _interval(row):
    _require(isinstance(row.get("start"), int) and isinstance(row.get("end"), int)
             and row["end"] >= row["start"], "missing/incomplete/invalid interval")


def _pid(row, field):
    _require(isinstance(row.get(field), int), f"missing {field}: cannot isolate PID")
    return row[field] >> 24


def _intersects(row, start, end):
    return row["start"] < end and row["end"] > start


def _phase(row, final):
    if row["end"] <= final["start"]:
        return "before_final"
    if row["start"] >= final["end"]:
        return "after_final"
    if final["start"] <= row["start"] and row["end"] <= final["end"]:
        return "inside_final"
    return "overlaps_final_boundary"


def _range_names(row, ranges, *, same_thread):
    return [{"name": item["name"], "start_ns": item["start"],
             "end_ns": item["end"], "global_tid": item["globalTid"]}
            for item in ranges
            if item["start"] <= row["start"] and row["end"] <= item["end"]
            and (not same_thread or item["globalTid"] == row["globalTid"])]


def _session_windows(ranges, generation_stages):
    """Pair FIFO GR output markers with same-thread Future/reconcile ranges."""
    by_pid = defaultdict(list)
    for row in ranges:
        by_pid[row["pid"]].append(row)
    windows = []
    for pid, items in by_pid.items():
        items.sort(key=lambda row: row["start"])
        boots = [r for r in items if _is(r, BOOTSTRAP)]
        finals = [r for r in items if _is(r, FINAL)]
        if not boots and not finals:
            _require(not any(_is(r, SCHEDULE) or _is(r, PROGRESS) for r in items),
                     f"PID {pid}: orphan GR markers / missing bootstrap")
            continue
        _require(len(boots) == len(finals), f"PID {pid}: missing bootstrap/final")
        schedules = [r for r in items if _is(r, SCHEDULE)]
        starts = []
        for boot in boots:
            preceding = [r for r in schedules if r["end"] <= boot["start"]]
            _require(bool(preceding), f"PID {pid}: missing prefill schedule")
            starts.append(preceding[-1]["start"])
        _require(len(set(starts)) == len(starts), f"PID {pid}: ambiguous prefill schedule")
        consumed = set()
        for index, (boot, start) in enumerate(zip(boots, starts)):
            limit = starts[index + 1] if index + 1 < len(starts) else float("inf")
            region = [r for r in items if start <= r["start"] < limit]
            outputs = [r for r in region if _is(r, PROGRESS) or _is(r, FINAL)]
            _require(len(outputs) == generation_stages and _is(outputs[-1], FINAL)
                     and all(_is(r, PROGRESS) for r in outputs[:-1]),
                     f"PID {pid} session {index}: incomplete/ambiguous output stages")
            final = outputs[-1]
            _require(boot["end"] <= outputs[0]["start"], "bootstrap/output order mismatch")
            stages = [r for r in region if _is(r, SCHEDULE) and r["start"] < final["end"]]
            _require(len(stages) == generation_stages,
                     f"PID {pid} session {index}: incorrect generation schedule count")
            _require(all(stage["end"] <= output["start"]
                         for stage, output in zip(stages, outputs)), "schedule/output order mismatch")
            reconciles = [r for r in region if _is(r, RECONCILE)]
            futures = [r for r in region if _is(r, FUTURE)]
            _require(len(reconciles) >= generation_stages and len(futures) == generation_stages,
                     f"PID {pid} session {index}: missing/extra Future or reconcile; ambiguous ownership")
            selected = reconciles[:generation_stages]
            tid = boot["globalTid"]
            _require(all(r["globalTid"] == tid for r in stages + outputs + selected + futures),
                     "cross-thread GR stages: unsupported/ambiguous ownership")
            previous_end = start
            for output, future, reconcile in zip(outputs, futures, selected):
                _require(previous_end <= future["start"] and output["end"] <= future["start"]
                         and future["end"] <= reconcile["start"],
                         "output/Future/reconcile FIFO order mismatch")
                parents = [r for r in items if _is(r, ENGINE) and r["globalTid"] == tid
                           and r["start"] <= future["start"] and reconcile["end"] <= r["end"]]
                _require(len(parents) == 1, "missing/ambiguous EngineCore Future/reconcile scope")
                previous_end = reconcile["end"]
            end = selected[-1]["end"]
            _require(end <= limit, "overlapping GR sessions")
            # Housekeeping after FIFO drain has no output-bearing Future and
            # is deliberately not counted as another generation completion.
            _require(all(r["start"] >= end for r in reconciles[generation_stages:]),
                     "overlapping housekeeping reconciliation")
            for row in outputs + [boot]:
                consumed.add(id(row))
            windows.append({"pid": pid, "start": start, "end": end, "final": final,
                            "reconciles": selected, "ranges": items,
                            "housekeeping_reconciles_after_session": len(reconciles) - generation_stages})
        _require(all(id(r) in consumed for r in items
                     if _is(r, BOOTSTRAP) or _is(r, PROGRESS) or _is(r, FINAL)),
                 "orphan GR output markers outside session windows")
    _require(bool(windows), "no complete GR sessions")
    return sorted(windows, key=lambda row: (row["start"], row["pid"]))


def analyze_trace(path, generation_stages):
    _require(type(generation_stages) is int and generation_stages > 0,
             "generation_stages must be an explicit positive integer")
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
        _require({"StringIds", "NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_MEMCPY"} <= tables,
                 "missing NVTX/string/memcpy capture tables")
        api_tables = [name for name in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER")
                      if name in tables]
        _require(bool(api_tables), "missing CUDA API capture tables")
        strings = dict(db.execute("select id,value from StringIds"))
        ranges = []
        for raw in db.execute("select * from NVTX_EVENTS order by start"):
            row = dict(raw)
            row["name"] = row.get("text") or strings.get(row.get("textId"), "")
            relevant = any(_is(row, name) for name in
                           (SCHEDULE, BOOTSTRAP, PROGRESS, FINAL, RECONCILE, FUTURE, ENGINE))
            if row.get("end") is None and not relevant:
                continue  # Unclosed capture annotations / instantaneous marks.
            _interval(row)
            row["pid"] = _pid(row, "globalTid")
            ranges.append(row)
        apis = []
        for table in api_tables:
            for raw in db.execute(f"select * from {table} order by start"):
                row = dict(raw)
                _interval(row)
                row["pid"] = _pid(row, "globalTid")
                _require(row["nameId"] in strings, "unknown CUDA API name")
                row["name"] = strings[row["nameId"]]
                row["table"] = table
                apis.append(row)
        copies = [dict(row) for row in db.execute("select * from CUPTI_ACTIVITY_KIND_MEMCPY")]
    for row in copies:
        _interval(row)
        row["pid"] = _pid(row, "globalPid")
        _require(type(row.get("bytes")) is int and row["bytes"] >= 0, "invalid copy bytes")
    windows = _session_windows(ranges, generation_stages)
    reports = []
    for index, window in enumerate(windows):
        pid, start, end, final = (window[key] for key in ("pid", "start", "end", "final"))
        local_ranges = window["ranges"]
        d2h = []
        for row in copies:
            if row["pid"] != pid or row["copyKind"] != 2 or not _intersects(row, start, end):
                continue
            launches = [api for api in apis if api["pid"] == pid
                        and api.get("correlationId") == row.get("correlationId")
                        and api["start"] <= row["start"]]
            # Preserve all matching runtime/driver records; do not pretend a
            # possibly reused correlation ID identifies one unique launch.
            latest = max((api["start"] for api in launches), default=None)
            launches = [api for api in launches if api["start"] == latest]
            d2h.append({"bytes": row["bytes"], "start_ns": row["start"], "end_ns": row["end"],
                        "duration_us": (row["end"] - row["start"]) / 1000,
                        "final_relation": _phase(row, final),
                        "crosses_session_boundary": row["start"] < start or row["end"] > end,
                        "device_time_containing_nvtx_ranges": _range_names(row, local_ranges, same_thread=False),
                        "launch_api_candidates": [{"name": api["name"], "start_ns": api["start"],
                                                   "nvtx_ranges": _range_names(api, local_ranges, same_thread=True)}
                                                  for api in launches]})
        controls = []
        for row in sorted(apis, key=lambda item: item["start"]):
            if (row["pid"] != pid or not _intersects(row, start, end)
                    or not any(word in row["name"].lower() for word in ("synchronize", "query"))):
                continue
            controls.append({"name": row["name"], "table": row["table"],
                             "start_ns": row["start"], "end_ns": row["end"],
                             "duration_us": (row["end"] - row["start"]) / 1000,
                             "return_value": row.get("returnValue"),
                             "final_relation": _phase(row, final),
                             "crosses_session_boundary": row["start"] < start or row["end"] > end,
                             "nvtx_ranges": _range_names(row, local_ranges, same_thread=True)})
        omitted_copies = [r for r in d2h if r["start_ns"] >= final["end"]]
        omitted_apis = [r for r in controls if r["start_ns"] >= final["end"]]
        reports.append({"session": index, "pid": pid, "start_ns": start, "end_ns": end,
                        "final_start_ns": final["start"], "final_end_ns": final["end"],
                        "generation_stages": generation_stages, "reconcile_count": len(window["reconciles"]),
                        "reconciles": [{"start_ns": r["start"], "end_ns": r["end"]} for r in window["reconciles"]],
                        "housekeeping_reconciles_after_session": window["housekeeping_reconciles_after_session"],
                        "d2h_count": len(d2h), "d2h_bytes": sum(r["bytes"] for r in d2h),
                        "d2h_summed_duration_us": sum(r["duration_us"] for r in d2h),
                        "d2h": d2h, "synchronize_query_apis": controls,
                        "api_counts": dict(Counter(r["name"] for r in controls)),
                        "omitted_by_final_end_start_filter": {
                            "d2h_count": len(omitted_copies),
                            "d2h_bytes": sum(r["bytes"] for r in omitted_copies),
                            "api_counts": dict(Counter(r["name"] for r in omitted_apis))}})
    return {"source": str(Path(path).resolve()), "generation_stages": generation_stages,
            "scope": "serial same-PID UniProc FIFO; prefill schedule through final reconciliation",
            "limitations": ["No request IDs: no mixed/concurrent request or cross-PID ownership inference.",
                            "D2H timing/size alone does not establish Beam/control payload identity.",
                            "Host API duration is not GPU stall time; marker coverage is not acceptance PASS.",
                            "Final relation classifies device copy/API intervals, not causality.",
                            "Duration sums may overlap; a boundary-crossing copy's bytes are not prorated."],
            "sessions": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite")
    parser.add_argument("--generation-stages", required=True, type=int)
    args = parser.parse_args()
    try:
        report = analyze_trace(args.sqlite, args.generation_stages)
    except (ValueError, sqlite3.Error) as error:
        parser.exit(2, f"trace rejected: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
