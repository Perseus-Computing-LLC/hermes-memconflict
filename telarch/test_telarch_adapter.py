"""Tests for telarch/eval_telarch.py. CPU only: no model, no network.

Run from the repository root:  python -m pytest telarch/test_telarch_adapter.py -q
"""
import json
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "benchmark"))

import eval_telarch as et  # noqa: E402
from eval_common import Build_Session_Dialogue_List  # noqa: E402

DATA = os.path.join(HERE, "..", "external", "MemConflict", "Data", "Step4_4.jsonl")
# Optional: path to the Hermes kibisis-vault plugin source (hermes-agent plugin) for the constants check.
PLUGIN_SRC = os.environ.get("KIBISIS_PLUGIN_SRC", "")
CONFIG = os.path.join(HERE, "memconflict-telarch-config.json")


def _persona(i=0):
    with open(DATA) as fh:
        for n, line in enumerate(fh):
            if n == i:
                return json.loads(line)
    raise IndexError(i)


def test_rounds_split_on_user_turns():
    d = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
         {"role": "assistant", "content": "c"}, {"role": "user", "content": "d"}]
    r = et.rounds_of(d)
    assert [[t["content"] for t in x] for x in r] == [["a", "b", "c"], ["d"]]
    assert et.round_text(r[0]) == "user: a\nassistant: b\nassistant: c"
    # A dialogue that opens with an assistant turn still yields one round per user turn after it.
    assert len(et.rounds_of([{"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}])) == 2


def test_created_at_is_session_noon_utc():
    assert et.created_at_for("2022-01-27") == "2022-01-27T12:00:00+00:00"
    assert et.created_at_for("") == "Unknown Time"
    assert et.created_at_for("Jan 2022") == "Jan 2022"


def test_reference_config_matches_preregistered_file():
    with open(CONFIG) as fh:
        reg = json.load(fh)
    assert reg["telarch_arm"]["settings"] == et.TELARCH_REFERENCE
    assert reg["telarch_arm"]["admission_gate"] is et.ADMISSION_GATE_PIPELINE


@pytest.mark.skipif(not PLUGIN_SRC or not os.path.exists(PLUGIN_SRC), reason="KIBISIS_PLUGIN_SRC not set")
def test_plugin_arm_constants_match_plugin_source():
    src = open(PLUGIN_SRC).read()
    body = src[src.index("def prefetch"):src.index("def queue_prefetch")]
    assert f"limit={et.PLUGIN_SEARCH_LIMIT}" in body
    assert f"hits[:{et.PLUGIN_RESULT_CAP}]" in body
    assert f"best / {et.PLUGIN_BEST_FRACTION_DIVISOR}" in body
    assert re.search(r'category="convention", scope="default"\)[^\n]*\n[^\n]*\n[^\n]*\]\[:4\]', body)
    assert f'scope="{et.PLUGIN_SCOPE}"' in body
    assert f'"match": "{et.PLUGIN_MATCH}"' in body
    # The remember tool's defaults the plugin arm writes with.
    assert 'auto_gate = bool(args.get("auto_gate", True))' in src
    assert 'scope = str(args.get("scope") or "default")' in src


def _ingest(binding, persona, n_sessions):
    ctx = binding.begin_persona(persona)
    questions = []
    for i, s in enumerate(persona["Full_Session_Chain"][:n_sessions]):
        meta = binding.ingest_session(ctx, s, Build_Session_Dialogue_List(s["Session_Dialogue"]), i)
        assert meta["Add_Duration_ms"] >= 0 and meta["Dialogue_Added_To_Memory"] > 0
        questions += [q["question"] for q in (s.get("Session_Questions") or [])]
    return ctx, questions


@pytest.mark.parametrize("arm", ["telarch", "kibisis-plugin"])
def test_arm_ingests_and_recalls(tmp_path, arm):
    persona = _persona(0)
    binding = et.ARMS[arm](str(tmp_path), 16, False)
    ctx, questions = _ingest(binding, persona, 8)
    assert questions
    try:
        for q in questions[:10]:
            items, ms = binding.recall(ctx, q, 5)
            assert ms >= 0
            for it in items:
                assert set(it) >= {"memory", "created_at", "score"}
                assert it["memory"].strip()
                assert re.match(r"^\d{4}-\d{2}-\d{2}T12:00:00\+00:00$", it["created_at"])
            if arm == "kibisis-plugin":
                assert len(items) <= et.PLUGIN_RESULT_CAP
                scores = [it["score"] for it in items]
                assert scores == sorted(scores, reverse=True)
            else:
                stats = ctx.get("_diag_provider_raw") or {}
                assert stats.get("context_tokens", 0) <= et.TELARCH_REFERENCE["budget"]
                assert stats.get("selected") == len(items)
                ranks = [it["relevance_rank"] for it in items]
                assert all(r is not None for r in ranks)
        counts = binding.persona_count_extras(ctx)
        tail = binding.persona_tail_extras(ctx)
        json.dumps(counts), json.dumps(tail)
        assert counts["Total_Recalls"] == min(10, len(questions))
    finally:
        binding.end_persona(ctx)
    assert not [p for p in os.listdir(tmp_path) if p.endswith(".sqlite")]


def test_telarch_items_are_chronological(tmp_path):
    persona = _persona(1)
    binding = et.ARMS["telarch"](str(tmp_path), 16, False)
    ctx, questions = _ingest(binding, persona, 12)
    try:
        items, _ = binding.recall(ctx, questions[-1], 5)
        assert items
        keys = [tuple(int(x) for x in it["id"].split(":")) for it in items]
        assert keys == sorted(keys)
    finally:
        binding.end_persona(ctx)


def test_plugin_gate_holds_some_rounds_out_of_recall(tmp_path):
    persona = _persona(0)
    binding = et.ARMS["kibisis-plugin"](str(tmp_path), 16, False)
    ctx, _ = _ingest(binding, persona, 8)
    try:
        assert ctx["puts"] == ctx["verified"] + ctx["candidates"]
        held = {e.key for e in ctx["vault"].list_entities(category=et.PLUGIN_CATEGORY, scope="default",
                                                           include_unverified=True)
                if str(getattr(e.epistemic_state, "value", e.epistemic_state)).lower() == "candidate"}
        assert len(held) == ctx["candidates"]
        if held:
            key = sorted(held)[0]
            text = ctx["vault"].get(et.PLUGIN_CATEGORY, key, scope="default").content
            items, _ = binding.recall(ctx, text[:200], 5)
            assert key not in {it["key"] for it in items}
    finally:
        binding.end_persona(ctx)
