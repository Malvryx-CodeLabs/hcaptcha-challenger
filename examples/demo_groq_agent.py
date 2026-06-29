"""
Solve hCaptcha using Groq's vision models instead of Gemini.

Setup:
    1. Create a Groq API key at https://console.groq.com
    2. export GROQ_API_KEY="gsk_..."   (or put it in a .env file)
    3. uv run playwright install --with-deps
    4. python examples/demo_groq_agent.py

Only two settings differ from demo_captcha_agent.py:
    - LLM_PROVIDER="groq"
    - GROQ_API_KEY

When LLM_PROVIDER='groq' and the model fields are left at their defaults,
they are automatically switched to Groq's most widely-available vision model:
    - classifier + reasoners -> meta-llama/llama-4-scout-17b-16e-instruct

A stronger model (e.g. meta-llama/llama-4-maverick-17b-128e-instruct) may solve
spatial challenges better, but it is not enabled on every account/tier and will
404 when unavailable. List what your key can access with:
    curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"
You can override any of them explicitly via the *_MODEL fields below.
"""
import asyncio
import json

from playwright.async_api import async_playwright, Page

from hcaptcha_challenger import AgentV, AgentConfig, CaptchaResponse
from hcaptcha_challenger.utils import SiteKey


async def challenge(page: Page) -> AgentV:
    """Automates the process of solving an hCaptcha challenge with Groq."""
    agent_config = AgentConfig(
        LLM_PROVIDER="groq",
        # GROQ_API_KEY is read from the environment by default; set it here to override:
        # GROQ_API_KEY="gsk_...",
        #
        # Optional explicit model overrides (otherwise Groq defaults are used):
        # IMAGE_CLASSIFIER_MODEL="meta-llama/llama-4-maverick-17b-128e-instruct",
        # SPATIAL_POINT_REASONER_MODEL="meta-llama/llama-4-maverick-17b-128e-instruct",
        # SPATIAL_PATH_REASONER_MODEL="meta-llama/llama-4-maverick-17b-128e-instruct",
        # CHALLENGE_CLASSIFIER_MODEL="meta-llama/llama-4-scout-17b-16e-instruct",
    )
    agent = AgentV(page=page, agent_config=agent_config)

    await agent.robotic_arm.click_checkbox()
    await agent.wait_for_challenge()

    return agent


# noinspection DuplicatedCode
async def main():
    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir="tmp/.cache/user_data",
            headless=False,
            locale="en-US",
        )

        page = await context.new_page()
        await page.goto(SiteKey.as_site_link(SiteKey.epic))

        agent: AgentV = await challenge(page)
        if agent.cr_list:
            cr: CaptchaResponse = agent.cr_list[-1]
            print(json.dumps(cr.model_dump(by_alias=True), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
