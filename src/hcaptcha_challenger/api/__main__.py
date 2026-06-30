# -*- coding: utf-8 -*-
"""
Entrypoint: ``python -m hcaptcha_challenger.api`` (or the ``hc-api`` console script).

Runs as a SINGLE process: the solver shares one browser and one in-memory queue,
so multiple web workers would each launch their own browser and break the global
concurrency limit. Scale out with more containers behind a load balancer.
"""
import asyncio

from loguru import logger

from hcaptcha_challenger.api.config import ApiSettings


def main() -> None:
    import uvicorn

    from hcaptcha_challenger.api.app import create_app

    settings = ApiSettings()
    settings.validate_security()  # fail fast before binding the port

    if settings.STREAM_ENABLED:
        _run_with_stream(settings)
        return

    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL,
        workers=1,
        access_log=True,
    )


def _run_with_stream(settings: ApiSettings) -> None:
    """Run the solver API and the MJPEG debug stream on two ports in one loop."""
    import uvicorn

    from hcaptcha_challenger.api.app import create_app
    from hcaptcha_challenger.api.stream import StreamHub, create_stream_app

    hub = StreamHub(
        fps=settings.STREAM_FPS,
        max_width=settings.STREAM_MAX_WIDTH,
        quality=settings.STREAM_QUALITY,
    )
    app = create_app(settings, stream_hub=hub)
    stream_app = create_stream_app(hub)

    main_server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=settings.HOST,
            port=settings.PORT,
            log_level=settings.LOG_LEVEL,
            workers=1,
            access_log=True,
        )
    )
    stream_server = uvicorn.Server(
        uvicorn.Config(
            stream_app,
            host=settings.HOST,
            port=settings.STREAM_PORT,
            log_level="warning",
            access_log=False,
        )
    )

    logger.success(
        f"Debug stream on http://{settings.HOST}:{settings.STREAM_PORT}/  "
        f"(VLC: open http://<host>:{settings.STREAM_PORT}/stream.mjpeg)"
    )

    async def _serve():
        await asyncio.gather(main_server.serve(), stream_server.serve())

    asyncio.run(_serve())


if __name__ == "__main__":
    main()
