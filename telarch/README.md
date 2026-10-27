# Telarch provider (Kibisis + Harpe)

Adapter for the shared MemConflict harness (contract v5) for the Telarch Suite:
an encrypted SQLCipher memory vault (`kibisis`) with keys-only BM25 recall, and a
deterministic context packer (`harpe`) with the Harpe ProfileLedger block. No LLM
or embedding model is called at ingest or recall; retrieval is CPU-only and
deterministic, so every answer-model call is one of the 3,750 question calls.

Rows contributed here are fixed-budget rows (8,000 and 3,000 o200k tokens of
memory context). We do not claim an unlimited-context number.

## Files

| Path | What it is |
|---|---|
| `eval_telarch.py` | The adapter. Arms: `telarch` (round-3 reference pipeline), `telarch-profile` (reference + ProfileLedger, 8k), `telarch-profile-compact` (same, 3k), `kibisis-plugin` (the Hermes kibisis-vault plugin recall path as of 2026-10-22, for the record; its constants are frozen at that date). |
| `memconflict-telarch-config.json` | Pre-registration of the reference arm's settings, written 2026-10-21 before any MemConflict run. |
| `test_telarch_adapter.py` | CPU-only tests (`python -m pytest telarch/test_telarch_adapter.py -q` from the repo root). |
| `Scores/telarch_v5ftc_gj12pen_eval_scores.jsonl` | Per-question `gj12pen` outputs, arm `telarch-profile`, run tag `v5ftc-telarchprofile-h100sxm`, harpe 0.1.0a8. **The provider row.** |
| `Scores/summary_v5ftc_gj12pen.json`, `manifest_v5ftc_gj12pen_score.json`, `serving_envelope_v5ftc_p0.json` | The harness's summary, score-stage manifest and serving envelope for that run. |
| `Scores/telarch_v5ftc_gj12pen_eval_scores_r4-blind.jsonl` | Per-question outputs of the earlier round-4 run, arm `telarch-profile-compact`, harpe 0.1.0a7. **The blind figure**; see below. |

## Results (macro answer accuracy, `gj12pen` judge, contract v5)

The ProfileLedger's attribute patterns were written against dev personas 0–7
(`Step4_4.jsonl` lines 1–8). Personas 8–29 (n = 2,744) are the held-out split.

| Run | Arm | harpe | All 30 personas (n = 3,750) macro / micro | Held-out 8–29 (n = 2,744) macro / micro | Held-out dynamic / static / conditional | Blind? |
|---|---|---|---|---|---|---|
| R5 (this PR's provider row) | `telarch-profile` (8k) | 0.1.0a8 | 0.851 / 0.851 | 0.843 / 0.848 | 0.848 / 0.739 / 0.943 | **No.** The 0.1.0a8 date/timeline parser fixes came out of diagnosing R4's misses on personas 8–29, so this is a targeted-improvement result. |
| R5 | `telarch-profile-compact` (3k) | 0.1.0a8 | 0.852 / 0.850 | 0.839 / 0.842 | 0.840 / 0.739 / 0.940 | No (same reason). |
| R4 (`…_r4-blind.jsonl`) | `telarch-profile-compact` (3k) | 0.1.0a7 | 0.818 / 0.789 | **0.804 / 0.779** | 0.764 / 0.701 / 0.946 | **Yes.** Tuned on 0–7 only; 8–29 never inspected before this run. |

Read the R5 row with the R4 row beside it. The R5 − R4 held-out gap on the same
2,744 questions is macro +0.039 (question-level bootstrap 95% interval
+0.020 … +0.059); the static-conflict component (+0.038, interval −0.010 … +0.088)
is not separable from noise. A third split (new personas, pre-registered
threshold, run once) is planned before any further rule change; until then
`harpe` 0.1.0a9 (which replays byte-identically except for 34 static-conflict
ledgers) stays unreleased.

Upstream's featured rows for context: Honcho 0.477 (0.482 on the held-out
2,744, recomputed from `honcho/Scores/honcho_v5ftc_gj12pen_eval_scores.jsonl`),
mem0 0.392, Supermemory 0.288, RetainDB server 0.270, Mnemosyne 0.116.

### Workload (upstream's formula, per dialogue turn)

`0.1 × cached + uncached + 5 × output`, per turn (÷ 71,060), from the vLLM
counters, as `benchmark/make_result_csvs.py` computes it:

| Arm | cached/turn | uncached/turn | output/turn | workload/turn |
|---|---|---|---|---|
| `telarch-profile` (8k) | 5.7 | 543.4 | 151.5 | 1,301.5 |
| `telarch-profile-compact` (3k) | 1.2 | 208.0 | 136.7 | 891.4 |

Both arms ran back-to-back on one vLLM server with non-overlapping windows.

## Pins

| Component | Version | Source |
|---|---|---|
| `harpe` | 0.1.0a8 | Release wheel <https://github.com/Perseus-Computing-LLC/harpe/releases/download/v0.1.0-alpha.8/harpe-0.1.0a8-py3-none-any.whl>, SHA-256 `5e0f4e8c8d62c0e2f9f53f0f94f45eacce8c73134334e1dca7a27c71854f1c66` (`SHA256SUMS` on the release). The run itself used a wheel built from commit `f264f4aff22fa0e3e2b174b9090800b8ad112904` (the 0.1.0a8 version bump); the tag `v0.1.0-alpha.8` (`295f72a0`) adds one test on top of it and no library code. |
| `kibisis` | 0.1.0a9 | Release wheel <https://github.com/Perseus-Computing-LLC/kibisis/releases/download/v0.1.0-alpha.9/kibisis-0.1.0a9-py3-none-any.whl>, SHA-256 `f1b9570478a872f2713acaa65f517359ee2bdf1e9d14a9cc03fc8b7f447262a4`; the same version is on PyPI as `kibisis==0.1.0a9`. |
| `telarch_evals` | `TelarchMemory` (the pipeline the adapter wraps) | <https://github.com/Perseus-Computing-LLC/telarch-benchmarks> commit `7ec0d0363b7b6877d28eb9fc1ea9631df72ffb52`, `evals/` subdirectory. |
| Harness | this repository's `benchmark/` at the commit the PR targets | run on our branch at `6a70e9cd5bcbe8009f6caa91e2cc54d55984a9c3` (see "Harness hooks" below) |
| Hardware | Lambda `gpu_1x_h100_sxm5` (NVIDIA H100 80 GB SXM5), run completed 2026-10-27T07:46:11Z | |
| Models | answer `qwen3.5-4b` (`AxionML/Qwen3.5-4B-NVFP4`, 131,072-token window), judge `gemma-4-12b` with the `gj12pen` penalty rubric, exactly as the contract-v5 scripts under `benchmark/` set them | `Scores/manifest_v5ftc_gj12pen_score.json`, `Scores/serving_envelope_v5ftc_p0.json` |

Install the Python side with:

```
pip install -r benchmark/requirements-lock.txt
pip install \
  "https://github.com/Perseus-Computing-LLC/harpe/releases/download/v0.1.0-alpha.8/harpe-0.1.0a8-py3-none-any.whl#sha256=5e0f4e8c8d62c0e2f9f53f0f94f45eacce8c73134334e1dca7a27c71854f1c66" \
  "kibisis==0.1.0a9" \
  "telarch_evals @ git+https://github.com/Perseus-Computing-LLC/telarch-benchmarks@7ec0d0363b7b6877d28eb9fc1ea9631df72ffb52#subdirectory=evals" \
  pytest
python -m pytest telarch/test_telarch_adapter.py -q
```

## Running

The adapter follows `eval_common.ProviderBinding`:

```
python telarch/eval_telarch.py --arm telarch-profile \
  --output_jsonl_path telarch/Results/telarch_results_v5ftc-telarchprofile-h100sxm.jsonl [common args]
python telarch/eval_telarch.py --arm telarch-profile-compact \
  --output_jsonl_path telarch/Results/telarch_results_v5ftc-telarchprofilecompact-h100sxm.jsonl [common args]
```

Each persona gets its own SQLCipher vault with a fresh random passphrase that
lives only in the process; the vault directory is deleted at `end_persona`.
Dialogue is stored one round at a time (a user turn and the assistant turns
that follow it) with `created_at` = the session date at 12:00 UTC; neither arm
ranks by the OS clock, so the run never preloads libfaketime.

### Harness hooks used by the banked run

The banked run used two small, default-preserving additions on our branch that
are **not** part of this PR, to keep the diff to one provider directory:

1. `eval_common.ProviderBinding.recorded_memories(retrieved)` — a hook that lets
   a provider whose answer context is in date order record `Retrieved_Memories`
   in relevance order instead (so SEH@K reads a relevance-ordered list). Default
   returns the list unchanged, so no other provider changes. `eval_telarch.py`
   overrides it on the profile arms; without the hook the override is simply
   never called and `Retrieved_Memories` is stored in date order (answers and
   `AA` are unaffected; `SEH@K` would read a date-ordered list).
2. `write_manifest.TEMPORAL_CAPABILITY["telarch"] = "native"`.

plus `benchmark/docker/Dockerfile.telarch-profile`, `entrypoint.telarch.sh` and
the matching `presets.sh` / `docker-compose.yml` entries. Happy to send those as
a follow-up PR if you want the Docker path in-tree.

## Provenance of the score files

The two `eval_scores.jsonl` files are byte-identical to the ones in our banked
results tarballs (R5: `results-h100sxm.tar.gz`, SHA-256
`eaae396a9f546d5ad993e5d8b41fd61881284de6a0192b49e216a9710d728dd9`). The run
records (shard logs, `pip freeze`, image IDs, vLLM logs) are banked privately;
we can open them or attach them on request.
