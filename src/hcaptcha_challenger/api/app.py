# -*- coding: utf-8 -*-
"""FastAPI application factory for the solver API."""
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Awaitable, Callable, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from loguru import logger

from hcaptcha_challenger.api.browser import BrowserManager
from hcaptcha_challenger.api.config import ApiSettings
from hcaptcha_challenger.api.queue import QueueFull, QueueTimeout, SolveManager
from hcaptcha_challenger.api.schemas import SolveRequest, SolveResponse, StatsResponse
from hcaptcha_challenger.api.security import RateLimiter, build_auth_dependency

# A solve callable: (sitekey, siteurl, rqdata) -> result dict.
SolveFunc = Callable[[str, str, Optional[str]], Awaitable[dict]]

_REPO_URL = "https://github.com/QIN2DIM/hcaptcha-challenger"


def create_app(
    settings: Optional[ApiSettings] = None,
    *,
    solve_func: Optional[SolveFunc] = None,
    stream_hub=None,
) -> FastAPI:
    """
    Build the app.

    Args:
        settings: API settings (defaults to environment-derived ApiSettings).
        solve_func: Inject a solver to bypass the real browser (used in tests).
            When provided, the browser/provider are not initialised.
        stream_hub: Optional StreamHub for the debug live stream. When provided,
            concurrency is forced to 1 and each solve is screencast to the hub.
    """
    settings = settings or ApiSettings()
    settings.validate_security()

    # The debug stream has a single shared frame buffer, so only one solve can be
    # streamed at a time. Force concurrency to 1 (but keep the configured queue).
    max_concurrent = settings.MAX_CONCURRENT_SOLVES
    if stream_hub is not None and max_concurrent != 1:
        logger.warning(
            "STREAM_ENABLED: forcing MAX_CONCURRENT_SOLVES from "
            f"{max_concurrent} to 1 (the debug stream supports one task)."
        )
        max_concurrent = 1

    manager = SolveManager(max_concurrent, settings.MAX_QUEUE)
    rate_limiter = RateLimiter(settings.RATE_LIMIT_RPM)
    browser_manager: Optional[BrowserManager] = None
    agent_config = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal browser_manager, agent_config
        if solve_func is None:
            # Lazy imports so test/stub mode never needs a browser or provider.
            from hcaptcha_challenger.agent.challenger import AgentConfig

            agent_config = AgentConfig()  # validates provider credentials
            browser_manager = BrowserManager(
                headless=settings.HEADLESS, args=settings.BROWSER_ARGS
            )
            await browser_manager.start()
            app.state.solve_func = _make_real_solve_func(
                browser_manager, agent_config, settings, stream_hub
            )
            logger.success(
                f"Solver API ready - provider={agent_config.LLM_PROVIDER.value} "
                f"concurrency={max_concurrent} queue={settings.MAX_QUEUE}"
                + (" stream=on" if stream_hub is not None else "")
            )
        else:
            app.state.solve_func = solve_func
            logger.info("Solver API ready (injected solve_func; browser disabled).")
        try:
            yield
        finally:
            if browser_manager is not None:
                await browser_manager.stop()

    app = FastAPI(
        title="hCaptcha Solver API",
        version="1.0.0",
        docs_url="/docs" if settings.ENABLE_DOCS else None,
        redoc_url="/redoc" if settings.ENABLE_DOCS else None,
        openapi_url="/openapi.json" if settings.ENABLE_DOCS else None,
        lifespan=lifespan,
    )
    app.state.manager = manager
    app.state.settings = settings

    if settings.CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.CORS_ORIGINS,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response

    auth = build_auth_dependency(settings, rate_limiter)

    @app.get("/", include_in_schema=False)
    async def home():
        return RedirectResponse(url=_REPO_URL)

    @app.get("/healthz")
    async def healthz():
        """Liveness: the process is up."""
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        """Readiness: browser is connected (real mode) or stub is wired."""
        ready = solve_func is not None or (
            browser_manager is not None and browser_manager.is_running
        )
        if not ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="not ready"
            )
        return {"status": "ready", **manager.stats()}

    @app.get("/v1/stats", response_model=StatsResponse)
    async def stats(_: str = Depends(auth)):
        return StatsResponse(**manager.stats())

    @app.post("/v1/solve", response_model=SolveResponse)
    async def solve(req: SolveRequest, _: str = Depends(auth)):
        started = time.perf_counter()

        async def factory() -> dict:
            return await app.state.solve_func(req.sitekey, req.siteurl, req.rqdata)

        try:
            result = await manager.submit(
                factory,
                wait_timeout=settings.QUEUE_WAIT_TIMEOUT_S,
                solve_timeout=settings.SOLVE_TIMEOUT_S,
            )
        except QueueFull:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Service at capacity. Retry later.",
                headers={"Retry-After": "5"},
            )
        except QueueTimeout:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Queued too long. Retry later.",
                headers={"Retry-After": "10"},
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Solve timed out."
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.exception(f"Unexpected solve error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal solve error.",
            )

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return SolveResponse(elapsed_ms=elapsed_ms, **result)

    return app


def _make_real_solve_func(
    browser_manager: BrowserManager, agent_config, settings: ApiSettings, stream_hub=None
) -> SolveFunc:
    from hcaptcha_challenger.api.solver import solve_captcha

    async def _solve(sitekey: str, siteurl: str, rqdata: Optional[str]) -> dict:
        async with browser_manager.context() as ctx:
            return await solve_captcha(
                ctx,
                agent_config,
                sitekey=sitekey,
                siteurl=siteurl,
                rqdata=rqdata,
                nav_timeout_s=settings.NAV_TIMEOUT_S,
                stream_hub=stream_hub,
            )

    return _solve
