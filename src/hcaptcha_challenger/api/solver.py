# -*- coding: utf-8 -*-
"""
solve_captcha - drive AgentV against a sitekey/siteurl and return the token.

To make the token bind to the caller's ``siteurl`` (hCaptcha tokens are
domain-scoped), we intercept the top-level document navigation and serve a
minimal page that embeds the hCaptcha widget for the given sitekey. The browser's
origin is therefore ``siteurl``, while hCaptcha's own assets
(js.hcaptcha.com / newassets.hcaptcha.com) load normally.
"""
import time
from urllib.parse import urlparse

from loguru import logger
from playwright.async_api import BrowserContext, Route

from hcaptcha_challenger.agent.challenger import AgentConfig, AgentV

# Minimal page hosting the hCaptcha checkbox for a given sitekey. api.js without
# `render=explicit` auto-renders any element with class "h-captcha".
_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>verify</title>
  <script src="https://js.hcaptcha.com/1/api.js" async defer></script>
</head>
<body>
  <form id="form" action="/" method="POST">
    <div class="h-captcha" data-sitekey="{sitekey}"{rqdata}></div>
  </form>
</body>
</html>"""

_GET_TOKEN_JS = """() => {
  const el = document.querySelector('[name=h-captcha-response]');
  return el && el.value ? el.value : '';
}"""


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


async def solve_captcha(
    ctx: BrowserContext,
    agent_config: AgentConfig,
    *,
    sitekey: str,
    siteurl: str,
    rqdata: str | None = None,
    nav_timeout_s: float = 60.0,
) -> dict:
    """
    Returns a dict: {success, token, expiration, error, raw}.

    Never raises for an unsolved captcha (returns success=False); raises only on
    unexpected/internal errors so the route can map them to 5xx.
    """
    started = time.perf_counter()
    target_host = _host(siteurl)
    rqdata_attr = f' data-rqdata="{rqdata}"' if rqdata else ""
    body = _PAGE_TEMPLATE.format(sitekey=sitekey, rqdata=rqdata_attr)

    page = await ctx.new_page()

    async def _route_handler(route: Route) -> None:
        request = route.request
        # Replace ONLY the top document for the target host; let hCaptcha assets
        # (different host) pass through untouched.
        if request.resource_type == "document" and _host(request.url) == target_host:
            await route.fulfill(status=200, content_type="text/html; charset=utf-8", body=body)
        else:
            await route.continue_()

    await ctx.route("**/*", _route_handler)

    try:
        await page.goto(
            siteurl, wait_until="domcontentloaded", timeout=int(nav_timeout_s * 1000)
        )

        agent = AgentV(page=page, agent_config=agent_config)
        await agent.robotic_arm.click_checkbox()
        await agent.wait_for_challenge()

        token = await page.evaluate(_GET_TOKEN_JS)
        cr = agent.cr_list[-1] if agent.cr_list else None

        success = bool(cr.is_pass) if cr is not None else bool(token)
        result = {
            "success": success,
            "token": token or (cr.generated_pass_UUID if cr else None) or None,
            "expiration": (cr.expiration if cr else None),
            "error": None if success else ((cr.error if cr else None) or "challenge_not_passed"),
            "raw": cr.model_dump(by_alias=True) if cr else None,
        }
        elapsed = int((time.perf_counter() - started) * 1000)
        logger.info(
            f"solve sitekey={sitekey[:8]}… host={target_host} "
            f"success={success} elapsed_ms={elapsed}"
        )
        return result
    finally:
        try:
            await page.close()
        except Exception:  # pragma: no cover
            pass
