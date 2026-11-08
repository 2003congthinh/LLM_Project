# Host an LLM: Step-by-Step Roadmap

> Read this from top to bottom. Each milestone builds on the one before it.
> Tick the boxes (`- [x]`) as you go.
> Prerequisite: you have a model to host. That can be your own GPT
> (`01_build_production_llm.md`, M0–M8) or a fine-tuned open-weight model (M9–M10).
> You can start H0–H2 right away with any open model from Hugging Face.

---

## 0. Read this first: what "hosting an LLM" means

Hosting means turning a file of weights into a **service** that many people
can call at the same time, quickly, safely, and at a cost you can afford.

```
User / App ──HTTPS──▶ API gateway ──▶ Inference server ──▶ GPU(s) holding the model
               (auth, rate limit,      (vLLM / SGLang /      (weights + KV cache
                logging, UI)            llama.cpp)            in VRAM)
```

Your `chatbot.py` is a **script**: one user, one terminal, the model reloaded
every run, and no way for another program to talk to it. By the end of this
roadmap, the same model will sit behind an HTTPS API that apps can call, with
streaming, monitoring, and autoscaling.

**The numbers you'll care about**

| Metric | Meaning | Typical target for chat |
|---|---|---|
| **TTFT** (time to first token) | How long until the user sees the first word | < 0.5–1 s |
| **TPOT / ITL** (time per output token) | Speed of the streaming text | < 50 ms (> 20 tokens/s feels smooth) |
| **Throughput** | Total tokens/sec across all users | As high as possible per GPU |
| **Concurrency** | How many requests are served at once | Depends on VRAM (KV cache) |
| **Cost per 1M tokens** | GPU $/hour ÷ tokens/hour | Compare against paid APIs |
| **p95 / p99 latency** | Latency of the slowest 5% / 1% of requests | Set as an SLO |

---

## Milestone overview

Tick a milestone only when every small goal inside it is done.

- [ ] **H0:** Understand how LLM inference works
- [ ] **H1:** Serve your own model with a hand-written API (FastAPI)
- [ ] **H2:** Run models locally with real tools (Ollama / llama.cpp)
- [ ] **H3:** Use a production inference engine (vLLM)
- [ ] **H4:** Containerize everything (Docker + GPU)
- [ ] **H5:** Build the application layer (auth, limits, chat UI)
- [ ] **H6:** Deploy to a cloud GPU with HTTPS
- [ ] **H7:** Monitor, benchmark, and load test
- [ ] **H8:** Scale, optimize cost, and stay reliable
- [ ] **H9:** Secure the service and handle abuse
- [ ] **H10:** Automate releases (CI/CD and the model lifecycle)

---

## H0: Understand how LLM inference works

**Goal:** Understand where the time and the memory go, because every hosting decision depends on it.

**Knowledge to learn**
- **Prefill vs. decode:** the prompt is processed in parallel (prefill, compute-bound). Then tokens are produced one at a time (decode, memory-bandwidth-bound). Your `generate()` loop is the decode phase.
- **KV cache:** store each layer's keys and values so earlier tokens aren't recomputed. It grows with *context length × concurrent users*.
- **VRAM math** (memorize this):
  - Weights = `parameters × bytes per parameter` (fp32 = 4, fp16/bf16 = 2, int8 = 1, int4 ≈ 0.5)
  - KV cache per token = `2 × n_layers × n_kv_heads × head_dim × bytes`
  - Total ≈ weights + KV cache × tokens in flight + ~10–20% overhead
- **Sampling:** temperature, top-k, top-p, repetition penalty, max tokens, stop sequences, seed
- **Batching:** static batching vs. **continuous batching**. Continuous batching is the main reason vLLM is so much faster than a naive loop.
- **Quantization for serving:** GGUF (CPU/GPU, llama.cpp), AWQ / GPTQ (GPU, vLLM), FP8 (newer GPUs)

**Worked example**

| Model | fp16 weights | int4 weights | KV cache per token (fp16) |
|---|---|---|---|
| Your `model-01` (~32M params) | ~64 MB (128 MB in fp32, matching your `.pkl` size) | — | tiny |
| 1B model | ~2 GB | ~0.7 GB | small |
| 8B Llama-style (32 layers, 8 KV heads, head_dim 128) | ~16 GB | ~5 GB | 2×32×8×128×2 = **128 KB** → 8k context = 1 GB per user |
| 70B model | ~140 GB | ~40 GB | larger |

On your **RTX 4060 (8 GB)**, an 8B model only fits at 4-bit. A 1B–4B model fits comfortably.

**Steps**
1. Read the Hugging Face docs page on text generation strategies (greedy, sampling, top-k/top-p, beam search).
2. Add temperature + top-k + top-p sampling to your own `generate()` and try different settings on the same prompt.
3. Time the prefill and decode phases separately in your own model.
4. Do the VRAM math for 3 models you might host and check it against `nvidia-smi`.
5. Read the vLLM / PagedAttention paper intro: *Efficient Memory Management for Large Language Model Serving with PagedAttention*.

**Small goals**
- [ ] Can explain prefill vs. decode and which one is compute-bound vs. memory-bound
- [ ] Can explain what the KV cache stores and why it limits concurrency
- [ ] Can calculate VRAM for weights + KV cache for any model
- [ ] Implemented temperature, top-k, and top-p sampling
- [ ] Can explain continuous batching in your own words
- [ ] Can explain the difference between GGUF, AWQ/GPTQ, and FP8

✅ **Done when:** given a model and a GPU, you can predict whether it fits and roughly how many users it can serve.

---

## H1: Serve your own model with a hand-written API

**Goal:** Turn `chatbot.py` into a web API. You'll see what inference servers do for you by doing it once yourself.

**Knowledge to learn**
- HTTP basics: methods, status codes, JSON, headers
- **FastAPI** + **Pydantic** (request validation) + **Uvicorn** (ASGI server)
- **Streaming** with Server-Sent Events (SSE), which is why ChatGPT text appears word by word
- Loading the model **once at startup** (not per request), `torch.inference_mode()`, `model.eval()`
- **Never load pickle in a server.** Use `state_dict` + `weights_only=True` or `safetensors` (see `01_build_production_llm.md`, M0).
- Concurrency problems: one GPU, many requests (a lock or queue for now; H3 solves this properly)

**Steps**
1. Create `server/app.py`:
   ```python
   import json, threading, torch
   from fastapi import FastAPI
   from fastapi.responses import StreamingResponse
   from pydantic import BaseModel, Field

   app = FastAPI()
   model, tokenizer = load_model("ckpt.pt")  # your loader: state_dict, NOT pickle
   gpu_lock = threading.Lock()                # one generation at a time (for now)

   class GenerateRequest(BaseModel):
       prompt: str = Field(min_length=1, max_length=4000)
       max_new_tokens: int = Field(default=200, ge=1, le=1000)
       temperature: float = Field(default=0.8, gt=0, le=2.0)

   @app.get("/health")
   def health():
       return {"status": "ok"}

   @app.post("/generate")
   def generate(req: GenerateRequest):
       def stream():
           with gpu_lock, torch.inference_mode():
               for text in generate_stream(model, tokenizer, req.prompt,
                                           req.max_new_tokens, req.temperature):
                   yield f"data: {json.dumps({'text': text})}\n\n"
           yield "data: [DONE]\n\n"
       return StreamingResponse(stream(), media_type="text/event-stream")
   ```
2. Write `generate_stream()` as a Python generator that `yield`s each new token as soon as it's sampled. Change your `generate()` loop so it `yield`s instead of returning at the end.
3. Run it with `uvicorn server.app:app --port 8000` and test with `curl -N` and the auto docs at `http://localhost:8000/docs`.
4. Write a tiny Python client that streams the response and prints tokens as they arrive.
5. Send 5 requests at the same time and watch them queue behind the lock. That's the problem continuous batching solves.

**Small goals**
- [ ] Model loads once at startup from `state_dict`/`safetensors` (no pickle)
- [ ] `/health` endpoint works
- [ ] `/generate` validates input with Pydantic (rejects empty or huge prompts)
- [ ] Streaming works: tokens appear one by one with `curl -N`
- [ ] Wrote a streaming Python client
- [ ] Observed and can explain what happens with concurrent requests

✅ **Done when:** another program on your machine can call your model over HTTP and receive streamed text.

---

## H2: Run models locally with real tools

**Goal:** Learn the tools people actually use to run LLMs on personal hardware, and the **OpenAI-compatible API** standard.

**Knowledge to learn**
- **Ollama** (easiest), **llama.cpp** (the engine underneath, more control), and the **GGUF** format
- Quantization levels: Q4_K_M (good default), Q5_K_M, Q8_0. Know the size vs. quality tradeoff.
- The **OpenAI-compatible API** (`/v1/chat/completions`, `/v1/completions`, `/v1/models`). Almost every tool and SDK speaks it, so if your server speaks it too, any client works.
- Chat templates at inference time (the server must format messages exactly as in training)
- **Open WebUI**: a ChatGPT-like web interface that connects to any OpenAI-compatible server
- Converting Hugging Face models to GGUF with llama.cpp's `convert_hf_to_gguf.py` + `llama-quantize`

**Steps**
1. Install Ollama on Windows, run a small model (for example a 3B–4B instruct model), and watch VRAM in `nvidia-smi`.
2. Call it through the OpenAI-compatible endpoint (`http://localhost:11434/v1`) with the official `openai` Python SDK (`base_url=...`).
3. Try `llama-server` from llama.cpp with a GGUF file. Change the context size, GPU layers (`-ngl`), and parallel slots, and measure tokens/sec.
4. Convert **your fine-tuned model from M9/M10** to GGUF, quantize it, and load it in Ollama with a `Modelfile` (set the chat template and stop tokens).
5. Run Open WebUI and connect it to your local server.
6. Note: your from-scratch GPT can't be converted unless its architecture matches one llama.cpp supports. That's why "make it Llama-compatible" in M4 is a useful stretch goal.

**Small goals**
- [ ] Ran an open model with Ollama and measured tokens/sec
- [ ] Called a local model with the `openai` Python SDK using `base_url`
- [ ] Ran `llama-server` and changed `-ngl`, context size, and parallel slots
- [ ] Converted a Hugging Face model to GGUF and quantized it (Q4_K_M + Q8_0)
- [ ] Your fine-tuned model runs in Ollama with the correct chat template
- [ ] Open WebUI connected to your local model
- [ ] Can explain why the OpenAI-compatible API matters

✅ **Done when:** you can chat with your own fine-tuned model in a web UI on your PC.

---

## H3: Use a production inference engine (vLLM)

**Goal:** Serve many users efficiently on GPUs. This is what companies run in production.

**Knowledge to learn**
- **vLLM:** PagedAttention, continuous batching, prefix caching, OpenAI-compatible server, serving LoRA adapters, quantized models (AWQ/GPTQ/FP8)
- **SGLang:** a strong alternative, especially for structured output and multi-turn caching
- When to use which engine: llama.cpp/Ollama for single users, CPUs, and laptops; vLLM/SGLang for multi-user GPU serving
- Key settings: `--max-model-len`, `--gpu-memory-utilization`, `--max-num-seqs`, `--quantization`, `--tensor-parallel-size`, `--enable-prefix-caching`
- **Structured output** (JSON schema / guided decoding) and **tool/function calling**
- **Linux requirement:** vLLM does not run natively on Windows. Use **WSL2** or Docker with GPU support.

**Steps**
1. Set up WSL2 (Ubuntu) with NVIDIA GPU support, and check that `nvidia-smi` works inside WSL.
2. Install vLLM in a Python virtual environment and serve a small model:
   `vllm serve <hf-model-id> --max-model-len 4096 --gpu-memory-utilization 0.9`
3. Call it with the `openai` SDK (base URL `http://localhost:8000/v1`), with and without streaming.
4. Serve **your M9 model**: either the merged weights, or base + LoRA adapter via `--enable-lora`.
5. Benchmark with vLLM's built-in benchmark tool and compare against your H1 FastAPI server and llama.cpp at 1, 8, and 32 concurrent users.
6. Try JSON-schema-constrained output for one request.

**Small goals**
- [ ] WSL2 + CUDA working, and `nvidia-smi` works inside Linux
- [ ] vLLM serving a model with an OpenAI-compatible API
- [ ] Your fine-tuned model (or LoRA adapter) served by vLLM
- [ ] Benchmarked H1 vs. llama.cpp vs. vLLM at several concurrency levels, results in a table
- [ ] Can explain what each key vLLM flag does
- [ ] Got valid JSON out using structured output

✅ **Done when:** your model serves 30+ concurrent streaming requests on one GPU and you have numbers that prove it.

---

## H4: Containerize everything (Docker + GPU)

**Goal:** Package the server so it runs the same way on your PC, a cloud VM, or a cluster.

**Knowledge to learn**
- Docker: images, containers, `Dockerfile`, layers, volumes, ports, environment variables
- GPU in containers: NVIDIA Container Toolkit, `docker run --gpus all`, Docker Desktop + WSL2 GPU support
- **Don't bake model weights into the image.** Mount them as a volume or download them at startup into a cache, because images with weights are huge and slow to ship.
- `docker compose` for multiple services (inference server + API gateway + UI)
- Secrets via environment variables / `.env` (never commit them), health checks, restart policies
- Image registries (Docker Hub, GitHub Container Registry)

**Steps**
1. Learn Docker basics with a non-GPU app first (containerize your H1 FastAPI server, CPU only).
2. Run the official vLLM Docker image with `--gpus all`, mounting your Hugging Face cache as a volume.
3. Write a `docker-compose.yml` with: `vllm` (model server), `api` (your gateway from H5), and `webui` (Open WebUI).
4. Add health checks and `restart: unless-stopped`.
5. Push your API image to a registry.

**Small goals**
- [ ] Containerized a CPU FastAPI app from scratch
- [ ] Ran vLLM in Docker with GPU access
- [ ] Model weights live in a volume, not inside the image
- [ ] `docker compose up` starts the whole stack (model + API + UI)
- [ ] Secrets come from `.env`, and `.env` is in `.gitignore`
- [ ] Image pushed to a container registry

✅ **Done when:** one command (`docker compose up`) brings up your whole chatbot stack on any machine with an NVIDIA GPU.

---

## H5: Build the application layer (auth, limits, chat UI)

**Goal:** Never expose the raw inference server to the internet. Put an API you control in front of it.

**Knowledge to learn**
- **API gateway pattern:** your FastAPI service sits between users and vLLM and forwards requests with `httpx` (async, streaming)
- **Authentication:** API keys (hashed in a database), or OAuth/JWT for a user-facing app
- **Rate limiting:** requests per minute *and* tokens per minute per user (for example with Redis)
- **Limits:** max prompt length, max output tokens, request timeouts, and cancelling generation when the client disconnects
- **Conversation management:** storing chat history (Postgres/SQLite), trimming history to fit the context window, system prompts
- **Usage tracking:** count input and output tokens per user (billing and abuse detection)
- Frontend: a simple chat UI (React/Next.js or plain HTML + JS reading SSE), CORS
- Optional: **RAG**: embeddings + a vector database (pgvector, Qdrant) to give the model your documents

**Steps**
1. Build `gateway/` with FastAPI: `POST /v1/chat/completions` that checks the API key, applies limits, and forwards a stream to vLLM.
2. Add per-key rate limiting with Redis.
3. Store conversations and token usage in a database.
4. Handle client disconnects: stop generation upstream so you don't waste GPU time.
5. Build a minimal chat web page (or point Open WebUI at your gateway).
6. Optional: add RAG over a small document set.

**Small goals**
- [ ] Gateway forwards streaming requests to vLLM
- [ ] Requests without a valid API key are rejected (401)
- [ ] Rate limits (requests/min and tokens/min) enforced (429)
- [ ] Prompt length, max tokens, and timeouts enforced
- [ ] Chat history stored and trimmed to fit the context window
- [ ] Token usage logged per user
- [ ] Client disconnects cancel generation
- [ ] A chat web page works end to end
- [ ] (Optional) RAG answers questions from your documents

✅ **Done when:** only authorized users can chat, nobody can hog the GPU, and you know how many tokens each user consumed.

---

## H6: Deploy to a cloud GPU with HTTPS

**Goal:** Put your stack on the internet.

**Knowledge to learn**
- **Hosting options:**

  | Option | Examples | Pros | Cons |
  |---|---|---|---|
  | Managed endpoints | Hugging Face Inference Endpoints, cloud vendors' model hosting | Fastest to set up | Less control, can cost more |
  | Serverless GPU | Modal, RunPod Serverless | Scale to zero, pay per second | Cold starts (loading the model takes time) |
  | Rented GPU VM | RunPod, Lambda, Vast.ai | Cheap, full control | You manage everything |
  | Hyperscaler | AWS, GCP, Azure | Enterprise features, compliance | Complex, often expensive, GPU quotas |
  | Your own PC | RTX 4060 + tunnel | Free | Not reliable, home network exposure |

- **Choosing a GPU** with your H0 VRAM math. Rough guide: 8 GB (your 4060) suits personal use of ≤ 8B at 4-bit. 24 GB (L4, A10G, RTX 4090) suits 8B in fp16 with some concurrency. 80 GB (A100/H100) suits 70B at 4-bit or 8B with high concurrency.
- Networking: domains, DNS, **HTTPS/TLS**, reverse proxies (**Caddy** for automatic HTTPS, or nginx), firewalls (only ports 80/443 open), SSH keys
- Model download at startup: Hugging Face token for private repos, caching on persistent disks
- Cost control: billing alerts, auto-shutdown, spot/interruptible instances

**Steps**
1. Start cheap: expose your **local** stack temporarily with a tunnel (for example Cloudflare Tunnel) to learn without paying.
2. Rent a 24 GB GPU VM. Install Docker + the NVIDIA toolkit, and run `docker compose up`.
3. Put Caddy in front with your domain, and confirm HTTPS works.
4. Close every port except 80/443 (and SSH with keys only).
5. Try one serverless deployment (Modal or RunPod Serverless) of the same model and measure cold-start time.
6. Set a billing alert and an auto-shutdown schedule.

**Small goals**
- [ ] Shared your local model through a tunnel
- [ ] Deployed the Docker stack on a rented cloud GPU
- [ ] Custom domain + HTTPS working
- [ ] Firewall: only 80/443 public, SSH uses keys only
- [ ] Tried a serverless GPU deployment and measured cold start
- [ ] Billing alert and auto-shutdown configured
- [ ] Wrote down the cost per hour and the estimated cost per 1M tokens

✅ **Done when:** a friend can use your chatbot from their phone over HTTPS, and you know exactly what it costs per hour.

---

## H7: Monitor, benchmark, and load test

**Goal:** Know when the service is slow, broken, or overloaded before your users tell you.

**Knowledge to learn**
- **Metrics:** Prometheus (vLLM exposes `/metrics`) + **Grafana** dashboards
- What to watch: TTFT, TPOT, request latency p50/p95/p99, tokens/sec, requests running/waiting, **KV cache usage %**, GPU utilization and memory (NVIDIA DCGM exporter), error rate
- **Logs:** structured JSON logs and request IDs; don't log sensitive user content by default
- **Tracing:** OpenTelemetry across gateway → vLLM
- **LLM-specific observability:** logging prompts and outputs (with consent) for quality review, user feedback (👍/👎)
- **Load testing:** Locust or k6, with realistic prompt and output lengths
- **SLOs and alerts:** for example "p95 TTFT < 1 s, error rate < 1%", alerting to email, Discord, or Telegram

**Steps**
1. Add Prometheus + Grafana to your compose file and scrape vLLM, your gateway, and GPU metrics.
2. Build a dashboard with the metrics listed above.
3. Write a Locust or k6 test with realistic chat traffic. Ramp up users until latency breaks your SLO, and record the max users per GPU.
4. Define 2–3 SLOs and set alerts for them.
5. Add a 👍/👎 button in the UI and store the feedback.

**Small goals**
- [ ] Prometheus scraping vLLM, gateway, and GPU metrics
- [ ] Grafana dashboard showing TTFT, TPOT, throughput, KV cache %, GPU utilization, and errors
- [ ] Structured logs with request IDs
- [ ] Load test written, and you know the max concurrent users per GPU
- [ ] SLOs defined, with alerts firing to your phone or email
- [ ] User feedback collected

✅ **Done when:** you can answer "how many users can one GPU handle, and how do I know when it's overloaded?" with data.

---

## H8: Scale, optimize cost, and stay reliable

**Goal:** Handle more users for less money, and survive failures.

**Knowledge to learn**
- **Horizontal scaling:** multiple replicas behind a load balancer; autoscaling on queue length or KV cache usage (not CPU)
- **Vertical scaling:** bigger GPU; **tensor parallelism** across GPUs for models that don't fit on one
- **Kubernetes** basics (Pods, Deployments, Services, GPU node pools); LLM-serving layers on top of it (for example KServe or vLLM's production stack)
- **Cold starts:** model loading time; keep warm replicas, use fast storage, cache weights
- **Cost optimization:**
  - Quantization (AWQ/FP8) to fit on a cheaper GPU or serve more users per GPU
  - Prefix caching for shared system prompts
  - Speculative decoding for faster decode
  - Right-sizing `max_model_len` (a smaller context leaves more room for concurrent users)
  - Spot instances, scale to zero at night
  - Routing: send easy requests to a small model and hard ones to a big one
- **Reliability:** health checks and readiness probes, retries with backoff, timeouts, graceful shutdown (finish in-flight streams), a fallback model or provider

**Steps**
1. Run 2 replicas behind a load balancer (Caddy/nginx) and kill one mid-stream to see what happens.
2. Measure the cost per 1M tokens for: fp16 vs. AWQ vs. FP8 (if your GPU supports it), with and without prefix caching.
3. Learn Kubernetes basics locally (kind/minikube without a GPU is fine for learning the concepts).
4. Implement graceful shutdown and a fallback route (if your model is down, return a clear error or route to a backup).
5. Write a short **capacity plan**: expected users → tokens/day → GPUs needed → monthly cost.

**Small goals**
- [ ] 2+ replicas behind a load balancer, survives killing one
- [ ] Autoscaling rule based on queue length or KV cache usage (designed or implemented)
- [ ] Cost per 1M tokens compared across quantization options
- [ ] Prefix caching enabled and its effect measured
- [ ] Learned Kubernetes basics (Deployment, Service, GPU scheduling concepts)
- [ ] Graceful shutdown + fallback implemented
- [ ] Capacity plan written

✅ **Done when:** you can double traffic without a redesign, and you know your monthly bill before it arrives.

---

## H9: Secure the service and handle abuse

**Goal:** Protect your users, your GPU bill, and your reputation.

**Knowledge to learn**
- **OWASP Top 10 for LLM Applications:** prompt injection, insecure output handling, sensitive data disclosure, model denial of service, supply-chain risks
- **Prompt injection:** treat model output as untrusted (never pass it straight to a shell, SQL, or HTML)
- **Model supply chain:** only load `safetensors` (pickle can run code); pin model revisions/hashes; don't use `trust_remote_code` on repos you haven't reviewed
- **Input/output moderation:** a guard model (for example Llama Guard) or a moderation classifier on both sides
- **Privacy:** PII in logs, data retention policy, user consent, encryption at rest and in transit; local laws (GDPR, Vietnam's personal data protection rules, and so on)
- **Abuse and cost attacks:** huge prompts, infinite-generation requests, credential stuffing on API keys
- Secrets management, least-privilege access, dependency updates

**Steps**
1. Read the OWASP Top 10 for LLM Applications and write down how each item applies to your app.
2. Add input and output moderation in the gateway.
3. Escape and sanitize model output before rendering it in HTML (block XSS through model output).
4. Audit your logs: remove or mask PII, and set a retention period.
5. Run your M10 red-team prompts against the **deployed** service (not just the model).
6. Try attacking your own rate limits and timeouts with oversized and endless requests.

**Small goals**
- [ ] OWASP LLM Top 10 reviewed against your app
- [ ] Only `safetensors` loaded, model revision pinned, no untrusted remote code
- [ ] Input + output moderation in place
- [ ] Model output safely escaped in the UI
- [ ] PII masked in logs, retention policy written
- [ ] Red-team prompts run against the live service
- [ ] Abuse tests (huge or endless requests) are blocked

✅ **Done when:** you've attacked your own service and fixed every hole you found.

---

## H10: Automate releases (CI/CD and the model lifecycle)

**Goal:** Ship new model versions safely and roll back in minutes.

**Knowledge to learn**
- **CI/CD:** GitHub Actions to lint, test, build images, and deploy
- **Model registry and versioning:** Hugging Face Hub tags, or MLflow; every deployment points to an exact model revision
- **Eval gate:** a new model must pass your eval suite (`eval.py` from M7) and your safety set before deploying
- **Release strategies:** blue-green, **canary** (send 5% of traffic to the new model first), A/B testing, shadow traffic
- **Rollback:** switch traffic back to the previous version with one command
- **Feedback loop:** user feedback + logged failures → new training data → next fine-tune (back to `01_build_production_llm.md`, M9)
- Infrastructure as code (Terraform), if you go further

**Steps**
1. Write tests for your gateway (auth, rate limit, request validation) and run them in GitHub Actions.
2. Build and push Docker images automatically on every merge to `main`.
3. Add an eval-gate job: download the candidate model, run evals, and fail the pipeline if scores drop.
4. Implement a canary release: route a small share of traffic to the new version and compare metrics and feedback.
5. Practice a rollback, and time it.
6. Write a runbook: what to do when latency spikes, the GPU dies, or the model says something harmful.

**Small goals**
- [ ] Tests run automatically on every push
- [ ] Docker images built and pushed by CI
- [ ] Each deployment pins an exact model version
- [ ] An eval gate blocks worse models from deploying
- [ ] Canary release done at least once
- [ ] Rollback practiced and takes under 5 minutes
- [ ] Runbook written
- [ ] Feedback → new training data loop documented

✅ **Done when:** you can release a new fine-tuned model on a Friday without fear.

---

## Final "production-ready hosting" checklist

- [ ] OpenAI-compatible API behind HTTPS, with auth and rate limits
- [ ] Served by a real inference engine (vLLM/SGLang) with continuous batching
- [ ] Everything containerized; one command to deploy
- [ ] Dashboards + alerts for TTFT, TPOT, errors, KV cache, GPU
- [ ] Load tested; max capacity per GPU known
- [ ] Cost per 1M tokens known, with billing alerts set
- [ ] Moderation, safe output rendering, PII-safe logs, safetensors only
- [ ] Health checks, graceful shutdown, fallback, rollback tested
- [ ] CI/CD with an eval gate and canary releases
- [ ] Runbook for incidents

---

## Learning resources

**Inference concepts**
- Hugging Face docs: *Text generation strategies*, *LLM inference optimization*
- Paper: *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM)
- Paper: *Fast Inference from Transformers via Speculative Decoding*

**Tools (read the official docs and quickstarts)**
- Ollama, llama.cpp (`llama-server`, GGUF conversion and quantization)
- vLLM (OpenAI-compatible server, engine arguments, LoRA, quantization, metrics)
- SGLang
- Open WebUI
- FastAPI (including `StreamingResponse`), Pydantic, httpx
- Docker, Docker Compose, NVIDIA Container Toolkit, WSL2 GPU support
- Prometheus, Grafana, Locust / k6
- Caddy (automatic HTTPS)
- Kubernetes basics (official tutorials)

**Security**
- OWASP Top 10 for Large Language Model Applications

**Books**
- Chip Huyen, *AI Engineering* (inference optimization, evaluation, and deployment of LLM apps)
- Chip Huyen, *Designing Machine Learning Systems* (production ML in general)
