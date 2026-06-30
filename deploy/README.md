# hCaptcha Solver API — Deployment

A production-ready HTTP service that solves hCaptcha challenges. Deploy it, then
`POST /v1/solve` with a `sitekey` and `siteurl`; it drives a headless Chromium
browser via [`AgentV`](../src/hcaptcha_challenger/agent/challenger.py) and returns
the hCaptcha token. Concurrency is bounded and excess requests are **queued**, so
the service stays within a small machine's resources.

> Built by **Malvryx CodeLabs** on top of [QIN2DIM/hcaptcha-challenger](https://github.com/QIN2DIM/hcaptcha-challenger).

## Quick start (Docker Compose)

```bash
cp deploy/.env.example deploy/.env
# edit deploy/.env: set a provider key (e.g. GEMINI_API_KEY) and HCAPTCHA_API_API_KEYS
docker compose -f deploy/docker-compose.yml up -d --build
```

Check it's up:

```bash
curl localhost:8000/healthz          # {"status":"ok"}
curl -H "Authorization: Bearer <your-key>" localhost:8000/v1/stats
```

Solve a challenge:

```bash
curl -X POST localhost:8000/v1/solve \
  -H "Authorization: Bearer <your-key>" \
  -H "Content-Type: application/json" \
  -d '{"sitekey":"a5f74b19-9e45-40e0-b45d-47ff91b7a6c2","siteurl":"https://example.com/login"}'
```

```json
{
  "success": true,
  "token": "P1_eyJ...",
  "expiration": 120,
  "elapsed_ms": 8400,
  "error": null,
  "raw": { "pass": true, "c": { "type": "hsw", "req": "..." } }
}
```

Submit the returned `token` to the target site as the `h-captcha-response`.

## Run without Docker

```bash
uv pip install -e ".[api]"
uv run playwright install --with-deps chromium
export GEMINI_API_KEY=...                       # a provider key
export HCAPTCHA_API_API_KEYS=change-me
hc-api                                           # or: python -m hcaptcha_challenger.api
```

## Endpoints

| Method | Path          | Auth | Description |
|--------|---------------|------|-------------|
| POST   | `/v1/solve`   | ✅   | Solve a challenge for `{sitekey, siteurl, rqdata?}`. |
| GET    | `/v1/stats`   | ✅   | Live `{active, queued, max_concurrent, max_queue}`. |
| GET    | `/healthz`    | —    | Liveness. |
| GET    | `/readyz`     | —    | Readiness (browser connected). |
| GET    | `/docs`       | —    | Swagger UI (toggle with `HCAPTCHA_API_ENABLE_DOCS`). |

### Backpressure / status codes

- `200` — solve attempted (`success` true/false in the body).
- `401` — missing/invalid API key.
- `422` — invalid `sitekey` (must be UUID) or `siteurl` (must be http(s)).
- `429` — rate limit hit, **or** the queue is full / waited too long (`Retry-After` set).
- `504` — the solve exceeded `SOLVE_TIMEOUT_S`.

## Capacity tuning (the important part)

Each concurrent solve runs **one isolated Chromium context (~300–500 MB)**. The
key knob is `HCAPTCHA_API_MAX_CONCURRENT_SOLVES`. Requests beyond it are queued
up to `MAX_QUEUE`; beyond that they get `429`.

| Machine                | `MAX_CONCURRENT_SOLVES` | `MAX_QUEUE` | Notes |
|------------------------|-------------------------|-------------|-------|
| **2 GB / 1 CPU**       | 1                       | 16          | Minimum viable. |
| **4 GB / 2 CPU** (default) | **2**               | **32**      | Ships as the default. |
| 8 GB / 4 CPU           | 4                       | 64          | |
| 16 GB / 8 CPU          | 8                       | 128         | |

Rules of thumb:
- Budget ~0.5 GB RAM per concurrent solve, plus ~0.5 GB for the app.
- Solving is mostly **network-bound** (waiting on the LLM), so concurrency can
  exceed CPU count — RAM is usually the limit, not CPU.
- Raise `SOLVE_TIMEOUT_S` on slow networks; lower `MAX_QUEUE` to shed load sooner.

## Scaling out

The service runs as a **single process** (one browser, one shared queue), so do
**not** add web workers. To scale, run more containers behind a load balancer;
each enforces its own concurrency/queue limits.

## Security checklist

- ✅ Set `HCAPTCHA_API_API_KEYS` (the service refuses to start with auth on and no keys).
- ✅ Terminate TLS at a reverse proxy (nginx/Caddy/Traefik) in front of the API.
- ✅ Keep `HCAPTCHA_API_CORS_ORIGINS` empty unless a browser app calls it directly.
- ✅ Consider disabling `/docs` in production (`HCAPTCHA_API_ENABLE_DOCS=false`).
- ✅ Provider API keys and `.env` are secrets — never commit them.
- ℹ️ Rate limiting is in-process (per container). For a global limit, enforce it
  at your reverse proxy / gateway.
