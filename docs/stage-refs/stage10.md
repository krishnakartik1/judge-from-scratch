# Stage 10 — Deployment recipes + benchmarks

**Prompt:** [`docs/claude-code-prompts.md` § Stage 10: Deployment recipes](../claude-code-prompts.md#stage-10-deployment-recipes)

**Scripts / artifacts:**
- `deployment/ollama/README.md` — local-deploy recipe. One-liner: `ollama run hf.co/krishnakartik/gemma4-social-bias-judge-gguf:Q8_0`.
- `deployment/vllm/Dockerfile` — production-pattern OpenAI-compatible API. bf16, `--max-model-len 4096`, `--gpu-memory-utilization 0.85`, no `--enforce-eager` (CUDA graphs on).
- `deployment/vllm/docker-compose.yml` — single-service compose for local testing.
- `deployment/vllm/README.md` — vLLM deploy guide + cloud-hosting notes.
- `deployment/example_client.py` — unified client (`--backend ollama|vllm`, `--benchmark` flag).
- `deployment/modal/vllm_bench.py` — throughput/latency benchmark on Modal A100.
- `deployment/benchmark_results.json` — captured numbers (committed to repo).

**Inputs:**
- Published HF artifacts from Stage 9.
- Stage 8 cached predictions at `/vol/eval/cache/{baseline,sft,dpo}_predictions.jsonl` (for offline batch numbers).

**Outputs (live numbers from `deployment/benchmark_results.json`):**

Offline batch (Stage 8 cache, 2,100 rows × 7 passes per pair on Modal A100-80GB):

| Model | Wall-clock (s) | Output tokens | prompts/min | Output tok/s |
|---|---|---|---|---|
| baseline | 330.4 | 210,808 | 381.34 | 638.0 |
| sft | 554.3 | 486,714 | 227.33 | 878.1 |
| dpo | 393.8 | 486,832 | 319.95 | 1,236.2 |

Live serving (vLLM, Modal A100-80GB, CUDA graphs **on**):

| Mode | n | p50 latency | p95 latency | tok/s |
|---|---|---|---|---|
| sequential (concurrency=1) | 50 | 2.16 s | 4.94 s | 73.0 |
| concurrent (concurrency=16) | 50 | 14.0 s | 19.9 s | 170.7 |

Live serving with `--enforce-eager` (Stage 8 parity flag — CUDA graphs **off**):

| Mode | tok/s |
|---|---|
| sequential | 24.9 |
| concurrent | 36.6 |

**The `--enforce-eager` finding:** turning off eager mode (i.e., letting vLLM's CUDA graphs capture and compile the inference kernel) yielded ~3× sequential and ~5× concurrent throughput on Gemma 4 E4B. Stage 8 used `--enforce-eager` for run-to-run determinism across baseline/SFT/DPO eval; production serving turns it off.

**Cost comparison (apples-to-apples, per-call):**

| Backend | USD per call | Source |
|---|---|---|
| Sonnet 4.6 labeling (Batch API) | $0.004229 | `data/labeled/.cost_ledger.jsonl` (primary + retry) |
| Self-hosted DPO judge (Modal A100-80GB) | $0.000130 | $2.50/hr ÷ 19,197 calls/hr = DPO offline batch throughput |
| **Ratio** | **32.47×** cheaper to self-host | |

(Per-pair-pipeline ratio is 8.12× — Sonnet pipeline includes cross-checkers the self-hosted judge doesn't run.)

**Decisions made:**
- [#15](../project-status.md#key-methodological-decisions-chronological) — Three deployment paths: Ollama (Stage 10) + vLLM (Stage 10) + HF Space (deferred to Stage 11). The Ollama path is the lowest-friction reader entry point.

**Key outputs:**
- Throughput speedup from CUDA graphs (3-5×) is the deployment-side educational finding. Stage 8 traded it for determinism; production traded determinism for it.
- The 32× cost ratio is the resume-line number — apples-to-apples per-judgment cost between Sonnet 4.6 labeling and the self-hosted judge.
