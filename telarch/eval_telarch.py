#!/usr/bin/env python3
"""Telarch Suite (Kibisis + Harpe) adapter for the shared MemConflict harness (contract v5).

Two arms, selected with --arm. Both keep one isolated, encrypted Kibisis vault per persona
(SQLCipher, a fresh random passphrase per persona that lives only in this process), and both
store the dialogue one round at a time: a user turn and the assistant turns that follow it,
the LongMemEval convention of telarch-benchmarks. Retrieval is CPU-only and deterministic;
no LLM or embedding model is called at ingest or recall.

--arm telarch (the featured arm: the Telarch pipeline)
  telarch_evals.memory.TelarchMemory from Perseus-Computing-LLC/telarch-benchmarks at the
  pinned commit, with the round-3 pre-registered reference settings (see
  telarch/memconflict-telarch-config.json): fanout BM25 query planner over Kibisis
  keys-only search, 200 candidates, 50 per term, Harpe least_relevant packing into an
  8,000-token budget counted in tiktoken o200k, a 0.9 fill floor, relative dates resolved
  against each round's session date (harpe resolve_relative_dates), conflict notes off.
  The vault takes each round as a verified entry (admission gate off), as the pipeline does
  on every other benchmark. The packed rounds go to the answer model in chronological order,
  the order Telarch hands its reader. plugin_native_recall=True: the harness hands over the
  whole Harpe-budgeted block, not a top-K slice (the ruling for providers that inject their
  own bounded block). Each item's `score` is its relevance (BM25 normalised to the best hit,
  0..1) and `relevance_rank` its place in the ranked candidates.

--arm kibisis-plugin (the deployed Hermes Kibisis plugin, recall surface)
  The plugin (hermes plugin kibisis-vault, KibisisMemoryProvider) has NO automatic write
  path: it stores memory only when the agent calls its kibisis_remember tool. This arm
  therefore writes each round the way that tool does with its defaults: vault.put() into
  scope "default" with the epistemic state left to the admission gate (auto_gate on, the
  tool's default), so the gate may hold a round as a `candidate`, which default recall does
  not return. The count is reported per persona (Total_Gated_Candidates). Recall replicates
  KibisisMemoryProvider.prefetch(): search(question, limit=12, scope="default", match="any"),
  drop confidential entries and conventions, keep hits whose BM25 rank_score is at most half
  the best one's (rank_score is negative; lower is better), take the first 6. Up to 4
  standing conventions are also injected by the plugin; a per-persona benchmark vault has
  none. plugin_native_recall=True: the harness hands over all of the (at most 6) items in
  plugin order. The plugin's own rendered "- [category/key] (state) text" block is not
  appended; the harness's shared rendering carries the same items. test_telarch_adapter.py
  diffs the recall constants against the plugin source.

Timestamps: every round carries its session's dataset date; `created_at` is that date at
noon UTC (the harness's logical_session_noon_v1 contract) in ISO 8601. Nothing reads the
wall clock, so BENCH_CLOCKSYNC is not needed and not used.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import secrets
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCHMARK_DIR = os.path.normpath(os.path.join(_HERE, "..", "benchmark"))
for _p in (_HERE, _BENCHMARK_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_common import (  # noqa: E402
    ProviderBinding,
    add_common_eval_args,
    record_provider_retrieval,
    run_eval,
)

# ---------------------------------------------------------------------------
# Round-3 reference settings (telarch/memconflict-telarch-config.json, written before any run).
# ---------------------------------------------------------------------------
TELARCH_REFERENCE: Dict[str, Any] = {
    "budget": 8000,
    "candidates": 200,
    "per_term": 50,
    "policy": "least_relevant",
    "encrypt": True,
    "counter": "o200k",
    "planner": "fanout",
    "fill_floor": 0.9,
    "passthrough_under": 0,
    "recency": 0.0,
    "mmr": 0.0,
    "bridge": 0,
    "resolve_dates": True,
    "surface_conflicts": False,
}
ADMISSION_GATE_PIPELINE = False

# ---------------------------------------------------------------------------
# Round-4 profile arm (--arm telarch-profile): the round-3 reference settings plus the Harpe
# ProfileLedger (harpe 0.1.0a7). NOT pre-registered: the ledger's attribute patterns were written
# against MemConflict dev personas 0-7 (Step4_4.jsonl lines 1-8), so a clean figure for this arm is
# the held-out personas 8-29; a 30-persona figure must disclose the dev overlap.
# ---------------------------------------------------------------------------
TELARCH_PROFILE: Dict[str, Any] = {**TELARCH_REFERENCE, "profile": True}
# V2 of the same arm: the ledger plus a 3,000-token context instead of 8,000. Round 3 showed the 4B reader
# turning an 8,000-token haystack into confident wrong answers (677 dynamic -1.0 verdicts). The variant to
# headline is chosen on the dev personas only (higher dev macro AA), before the held-out rows are read.
TELARCH_PROFILE_COMPACT: Dict[str, Any] = {**TELARCH_PROFILE, "budget": 3000}
PROFILE_DEV_PERSONAS = (0, 8)  # [start, end) persona indexes the ledger patterns were developed on

# ---------------------------------------------------------------------------
# Plugin-parity constants (KibisisMemoryProvider.prefetch). test_telarch_adapter.py checks
# them against the plugin source so the arm cannot drift from the plugin.
# ---------------------------------------------------------------------------
PLUGIN_SEARCH_LIMIT = 12
PLUGIN_RESULT_CAP = 6
PLUGIN_SCOPE = "default"
PLUGIN_MATCH = "any"
PLUGIN_BEST_FRACTION_DIVISOR = 2  # keep r.rank_score <= best / 2
PLUGIN_CATEGORY = "conversation"

RANKED_DIAG_CAP = 20  # relevance-ranked candidates kept per question as provenance
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def created_at_for(date: str) -> str:
    date = (date or "").strip()
    if _DATE.match(date):
        return f"{date}T12:00:00+00:00"
    return date or "Unknown Time"


def rounds_of(dialogue: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """One round per user turn plus the assistant turns that follow it."""
    rounds: List[List[Dict[str, Any]]] = []
    for turn in dialogue:
        if str(turn.get("role", "")) == "user" or not rounds:
            rounds.append([])
        rounds[-1].append(turn)
    return rounds


def round_text(turns: List[Dict[str, Any]]) -> str:
    return "\n".join(f"{t.get('role', 'user')}: {str(t.get('content', '')).strip()}" for t in turns)


def persona_id(persona_item: Dict[str, Any]) -> str:
    for key in ("ID", "Persona_ID", "persona_id", "id"):
        if persona_item.get(key) not in (None, ""):
            return str(persona_item[key])
    return hashlib.sha256(repr(sorted(persona_item.keys())).encode()).hexdigest()[:12]


def _summary(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 2),
        "p50": ordered[len(ordered) // 2],
        "max": ordered[-1],
    }


def make_ranked_memory(workdir: Path, **kwargs: Any):
    """TelarchMemory that also keeps the ranking its last recall packed from (diagnostic only).

    recall() may widen retrieval and rank again; the ranking kept is the one of the final pass,
    which is the one Harpe packed. Nothing about selection changes.
    """
    from telarch_evals.memory import TelarchMemory

    class RankedTelarchMemory(TelarchMemory):
        def _rank(self, history_id, scores, by_uid, limit):
            ranked = super()._rank(history_id, scores, by_uid, limit)
            self.last_ranked = list(ranked)
            return ranked

    memory = RankedTelarchMemory(workdir, **kwargs)
    memory.last_ranked = []
    return memory


class TelarchPipelineBinding(ProviderBinding):
    memory_system = "telarch"
    store_id_key = "Telarch_Store_ID"
    runtime_summary_key = "Telarch_Runtime_Summary"
    stage_name = "telarch_answer_generation"
    stage_note = "Telarch (Kibisis + Harpe) retrieval and question answering"
    plugin_native_recall = True
    settings: Dict[str, Any] = TELARCH_REFERENCE

    def __init__(self, workdir: str, cache_mb: Optional[int], keep_dbs: bool) -> None:
        from telarch_evals.memory import Unit  # noqa: F401 - fail early if the pin is missing

        self.workdir = Path(workdir)
        self.keep_dbs = keep_dbs
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.memory = make_ranked_memory(self.workdir, admission_gate=ADMISSION_GATE_PIPELINE,
                                         cache_mb=cache_mb, **self.settings)

    def config(self) -> Dict[str, Any]:
        return self.memory.config()

    def begin_persona(self, persona_item: Dict[str, Any]) -> Dict[str, Any]:
        pid = persona_id(persona_item)
        info = self.memory.open(pid, [])
        return {
            "store_id": "telarch-" + hashlib.sha256(pid.encode("utf-8")).hexdigest()[:20],
            "history_id": pid,
            "units_added": 0,
            "ingest_errors": int(info.get("ingest_errors", 0)),
            "context_tokens": [],
            "selected": [],
            "fill": [],
            "dated_units": 0,
            "widened": 0,
            "recalls": 0,
            "profile_blocks": 0,
            "profile_tokens": [],
        }

    def ingest_session(self, ctx: Dict[str, Any], session_item: Dict[str, Any],
                       dialogue: List[Dict[str, Any]], session_index: int) -> Dict[str, Any]:
        from telarch_evals.memory import Unit

        date = str(session_item.get("Date", "") or "")
        sid = str(session_item.get("Session_ID", session_index))
        units = []
        for ri, turns in enumerate(rounds_of(dialogue)):
            text = round_text(turns)
            units.append(Unit(uid=f"{session_index}:{ri}", session=sid, order=(session_index, ri), date=date,
                              text=text, rendered=f"[{date}]\n{text}",
                              turns=[{"role": t.get("role", "user"), "content": str(t.get("content", ""))}
                                     for t in turns]))
        t0 = time.perf_counter()
        info = self.memory.add(ctx["history_id"], units)
        add_ms = (time.perf_counter() - t0) * 1000.0
        ctx["units_added"] += int(info.get("units", 0))
        ctx["ingest_errors"] += int(info.get("ingest_errors", 0))
        return {
            "Add_Duration_ms": add_ms,
            "Dialogue_Added_To_Memory": int(info.get("units", 0)),
            "Rounds_In_Session": len(units),
            "Ingest_Errors": int(info.get("ingest_errors", 0)),
        }

    def recall(self, ctx: Dict[str, Any], question_text: str, top_k: int) -> Tuple[List[Dict[str, Any]], float]:
        t0 = time.perf_counter()
        rec = self.memory.recall(ctx["history_id"], question_text)
        search_ms = (time.perf_counter() - t0) * 1000.0
        ranked = list(self.memory.last_ranked or [])
        relevance = dict(ranked)
        rank_of = {uid: i for i, (uid, _) in enumerate(ranked)}
        items = []
        for unit in rec.units:
            items.append({
                "memory": unit.text,
                "created_at": created_at_for(unit.date),
                "score": round(float(relevance.get(unit.uid, 0.0)), 6),
                "id": unit.uid,
                "session": unit.session,
                "relevance_rank": rank_of.get(unit.uid),
                "dated": bool(unit.meta.get("dated")),
                "profile": bool(unit.meta.get("profile")),
            })
        _, _, by_uid = self.memory._open[ctx["history_id"]]
        ranked_items = []
        for uid, rel in ranked[:RANKED_DIAG_CAP]:
            unit = by_uid.get(uid)
            if unit is None:
                continue
            ranked_items.append({"memory": unit.text, "created_at": created_at_for(unit.date),
                                 "score": round(float(rel), 6), "id": uid})
        stats = dict(rec.stats)
        record_provider_retrieval(ctx, raw=stats, ranked=ranked_items)
        ctx["recalls"] += 1
        ctx["context_tokens"].append(int(stats.get("context_tokens_o200k", stats.get("context_tokens", 0)) or 0))
        ctx["selected"].append(int(stats.get("selected", len(items)) or 0))
        if stats.get("fill") is not None:
            ctx["fill"].append(float(stats["fill"]))
        ctx["dated_units"] += int(stats.get("dated_units", 0) or 0)
        ctx["widened"] += int(stats.get("widened", 0) or 0)
        if stats.get("profile_tokens"):
            ctx["profile_blocks"] += 1
            ctx["profile_tokens"].append(int(stats["profile_tokens"]))
        return items, search_ms

    def end_persona(self, ctx: Dict[str, Any]) -> None:
        if not self.keep_dbs:
            self.memory.close(ctx["history_id"])

    def persona_count_extras(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {"Total_Units_Added": ctx["units_added"], "Total_Ingest_Errors": ctx["ingest_errors"],
                "Total_Recalls": ctx["recalls"]}

    def persona_tail_extras(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "Context_Tokens_o200k": _summary(ctx["context_tokens"]),
            "Selected_Units": _summary(ctx["selected"]),
            "Budget_Fill": _summary(ctx["fill"]),
            "Total_Dated_Units_Injected": ctx["dated_units"],
            "Total_Widenings": ctx["widened"],
            "Total_Profile_Blocks": ctx["profile_blocks"],
            "Profile_Block_Tokens_o200k": _summary(ctx["profile_tokens"]),
            "Telarch_Config": self.memory.config(),
        }


class TelarchProfileBinding(TelarchPipelineBinding):
    """--arm telarch-profile: the round-3 pipeline plus the Harpe ProfileLedger (see TELARCH_PROFILE)."""

    memory_system = "telarch-profile"
    stage_note = "Telarch (Kibisis + Harpe, ProfileLedger) retrieval and question answering"
    settings = TELARCH_PROFILE

    def recorded_memories(self, retrieved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # The reader's pack stays in date order (profile block first); the stored list puts the profile
        # block first and then the units by relevance, so SEH@K scores the most relevant units rather than
        # the oldest ones (round 3 recorded date order, which on update questions are mostly stale).
        def key(item: Dict[str, Any]) -> Tuple[int, int, int]:
            rank = item.get("relevance_rank")
            return (0 if item.get("profile") else 1, 0 if rank is not None else 1, rank or 0)
        return sorted(retrieved, key=key)


class TelarchProfileCompactBinding(TelarchProfileBinding):
    """--arm telarch-profile-compact: the profile arm with a 3,000-token budget (TELARCH_PROFILE_COMPACT)."""

    memory_system = "telarch-profile-compact"
    stage_note = "Telarch (Kibisis + Harpe, ProfileLedger, 3k budget) retrieval and question answering"
    settings = TELARCH_PROFILE_COMPACT


class KibisisPluginBinding(ProviderBinding):
    memory_system = "kibisis-plugin"
    store_id_key = "Kibisis_Store_ID"
    runtime_summary_key = "Kibisis_Plugin_Runtime_Summary"
    stage_name = "kibisis_plugin_answer_generation"
    stage_note = "Kibisis Hermes plugin recall and question answering"
    plugin_native_recall = True

    def __init__(self, workdir: str, cache_mb: Optional[int], keep_dbs: bool) -> None:
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.cache_mb = cache_mb
        self.keep_dbs = keep_dbs
        for name in ("KIBISIS_PASSPHRASE", "KIBISIS_PASSPHRASE_FILE"):
            os.environ.pop(name, None)

    @staticmethod
    def _remove(path: Path) -> None:
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(str(path) + suffix).unlink(missing_ok=True)

    def begin_persona(self, persona_item: Dict[str, Any]) -> Dict[str, Any]:
        from kibisis import SQLiteMemoryVault

        pid = persona_id(persona_item)
        digest = hashlib.sha256(pid.encode("utf-8")).hexdigest()[:20]
        path = self.workdir / f"kibisis-plugin-{digest}.sqlite"
        self._remove(path)
        extra = {"cache_mb": self.cache_mb} if self.cache_mb else {}
        # The plugin's vault: encrypted (its default), admission gate at the vault default (on).
        vault = SQLiteMemoryVault(str(path), passphrase=secrets.token_urlsafe(32), **extra)
        return {"store_id": f"kibisis-plugin-{digest}", "vault": vault, "path": path, "dates": {},
                "puts": 0, "put_errors": 0, "candidates": 0, "verified": 0, "recalls": 0,
                "injected": [], "filtered_by_half_best": 0, "searched_hits": []}

    def ingest_session(self, ctx: Dict[str, Any], session_item: Dict[str, Any],
                       dialogue: List[Dict[str, Any]], session_index: int) -> Dict[str, Any]:
        vault = ctx["vault"]
        date = str(session_item.get("Date", "") or "")
        added = errors = gated = 0
        t0 = time.perf_counter()
        for ri, turns in enumerate(rounds_of(dialogue)):
            text = round_text(turns)
            if not text.strip():
                continue
            key = f"s{session_index:04d}-r{ri:03d}"
            try:
                # kibisis_remember with its defaults: scope "default", epistemic state left to the
                # admission gate (auto_gate True), no sensitivity.
                entity = vault.put(category=PLUGIN_CATEGORY, key=key, content=text,
                                   metadata={"date": date, "session": str(session_item.get("Session_ID", ""))},
                                   scope=PLUGIN_SCOPE, epistemic_state=None, auto_gate=True)
            except Exception:  # noqa: BLE001 - counted; a failed write is a lost memory, as in production
                errors += 1
                continue
            added += 1
            state = str(getattr(entity.epistemic_state, "value", entity.epistemic_state) or "").lower()
            if state == "candidate":
                gated += 1
            ctx["dates"][key] = date
        add_ms = (time.perf_counter() - t0) * 1000.0
        ctx["puts"] += added
        ctx["put_errors"] += errors
        ctx["candidates"] += gated
        ctx["verified"] += added - gated
        return {"Add_Duration_ms": add_ms, "Dialogue_Added_To_Memory": added, "Put_Errors": errors,
                "Gated_Candidates": gated}

    def recall(self, ctx: Dict[str, Any], question_text: str, top_k: int) -> Tuple[List[Dict[str, Any]], float]:
        vault = ctx["vault"]
        query = question_text.strip()
        t0 = time.perf_counter()
        # KibisisMemoryProvider.prefetch(): standing conventions first (a benchmark vault has none) ...
        conventions = [e for e in vault.list_entities(category="convention", scope=PLUGIN_SCOPE)
                       if str((e.metadata or {}).get("sensitivity", "")).lower() != "confidential"][:4]
        # ... then any-word FTS search, non-confidential, non-convention ...
        hits = [r for r in vault.search(query, limit=PLUGIN_SEARCH_LIMIT, scope=PLUGIN_SCOPE, match=PLUGIN_MATCH)
                if str((r.entity.metadata or {}).get("sensitivity", "")).lower() != "confidential"
                and r.entity.category != "convention"]
        searched = len(hits)
        # ... keeping hits at least half as good as the best (BM25 rank_score: lower is better).
        if hits:
            best = hits[0].rank_score
            hits = [r for r in hits if r.rank_score <= best / PLUGIN_BEST_FRACTION_DIVISOR]
        kept = len(hits)
        hits = hits[:PLUGIN_RESULT_CAP]
        search_ms = (time.perf_counter() - t0) * 1000.0
        items: List[Dict[str, Any]] = []
        for c in conventions:
            items.append({"memory": c.content, "created_at": "Unknown Time", "score": None,
                          "category": c.category, "key": c.key})
        for r in hits:
            e = r.entity
            date = str((e.metadata or {}).get("date", "") or ctx["dates"].get(e.key, ""))
            items.append({"memory": e.content, "created_at": created_at_for(date),
                          "score": round(-float(r.rank_score), 6), "category": e.category, "key": e.key,
                          "epistemic_state": str(getattr(e.epistemic_state, "value", e.epistemic_state))})
        record_provider_retrieval(ctx, raw={"searched": searched, "kept_half_best": kept,
                                            "injected": len(items), "conventions": len(conventions)})
        ctx["recalls"] += 1
        ctx["injected"].append(len(items))
        ctx["searched_hits"].append(searched)
        ctx["filtered_by_half_best"] += searched - kept
        return items, search_ms

    def end_persona(self, ctx: Dict[str, Any]) -> None:
        vault = ctx.get("vault")
        if vault is not None:
            vault.close()
        if not self.keep_dbs and ctx.get("path") is not None:
            self._remove(ctx["path"])

    def persona_count_extras(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {"Total_Remember_Calls_OK": ctx["puts"], "Total_Remember_Errors": ctx["put_errors"],
                "Total_Gated_Candidates": ctx["candidates"], "Total_Verified_Entries": ctx["verified"],
                "Total_Recalls": ctx["recalls"]}

    def persona_tail_extras(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {"Injected_Items": _summary(ctx["injected"]), "Search_Hits": _summary(ctx["searched_hits"]),
                "Total_Dropped_By_Half_Best_Filter": ctx["filtered_by_half_best"]}


ARMS = {"telarch": TelarchPipelineBinding, "telarch-profile": TelarchProfileBinding,
        "telarch-profile-compact": TelarchProfileCompactBinding, "kibisis-plugin": KibisisPluginBinding}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Telarch Suite (Kibisis + Harpe) MemConflict adapter")
    add_common_eval_args(
        parser,
        default_input_jsonl_path=os.path.join(_HERE, "..", "external", "MemConflict", "Data", "Step4_4.jsonl"),
        default_output_jsonl_path=os.path.join(_HERE, "Results", "telarch_results.jsonl"),
        default_output_json_path=os.path.join(_HERE, "Results", "telarch_results.json"),
        top_k_help="Shared top-K (recorded; both arms inject their own bounded block).",
        default_start_idx=0,
        default_end_idx=None,
    )
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--vault-dir", default=os.environ.get("TELARCH_VAULT_DIR") or "",
                        help="Scratch dir for the per-persona vaults (default: a temp dir).")
    parser.add_argument("--cache-mb", type=int, default=int(os.environ.get("TELARCH_CACHE_MB", "64")),
                        help="SQLite page cache per vault in MiB (speed only; results do not depend on it).")
    parser.add_argument("--keep-dbs", action="store_true", help="Keep the per-persona vaults after the run.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    vault_dir = args.vault_dir or tempfile.mkdtemp(prefix=f"telarch-{args.arm}-")
    os.makedirs(vault_dir, exist_ok=True)
    binding = ARMS[args.arm](vault_dir, args.cache_mb or None, args.keep_dbs)
    try:
        ok = run_eval(
            binding,
            input_jsonl_path=args.input_jsonl_path,
            output_jsonl_path=args.output_jsonl_path,
            output_json_path=args.output_json_path,
            top_k=args.top_k,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            max_sessions=args.max_sessions,
            max_questions_per_session=args.max_questions_per_session,
            overwrite_existing_answers=args.overwrite_existing_answers,
        )
    finally:
        if not args.vault_dir and not args.keep_dbs:
            shutil.rmtree(vault_dir, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
