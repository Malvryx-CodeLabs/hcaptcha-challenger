# -*- coding: utf-8 -*-
"""
Entrypoint: ``python -m hcaptcha_challenger.api`` (or the ``hc-api`` console script).

Runs as a SINGLE process: the solver shares one browser and one in-memory queue,
so multiple web workers would each launch their own browser and break the global
concurrency limit. Scale out with more containers behind a load balancer.
"""
from hcaptcha_challenger.api.config import ApiSettings


def main() -> None:
    import uvicorn

    from hcaptcha_challenger.api.app import create_app

    settings = ApiSettings()
    settings.validate_security()  # fail fast before binding the port
    app = create_app(settings)

    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL,
        workers=1,
        access_log=True,
    )


if __name__ == "__main__":
    main()
