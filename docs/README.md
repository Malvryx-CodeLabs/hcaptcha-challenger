# Documentation

## Modifications by Malvryx CodeLabs

> This is a modified fork. The following changes were made by **Malvryx CodeLabs** on top of the upstream [QIN2DIM/hcaptcha-challenger](https://github.com/QIN2DIM/hcaptcha-challenger) project.

- **Added Groq as an alternative LLM provider** alongside the default Gemini backend, so challenges can be solved with Groq's vision models (Llama 4 Scout / Maverick).
  - New `GroqProvider` (`src/hcaptcha_challenger/tools/internal/providers/groq.py`) talking to Groq's OpenAI-compatible endpoint via `httpx` — no new required dependencies.
  - New `LLM_PROVIDER` and `GROQ_API_KEY` settings on `AgentConfig`; the active provider only requires its own API key.
  - When `LLM_PROVIDER="groq"`, default model names auto-switch to Groq's vision models (overridable per task).
  - `LLMProvider` enum, `GroqModelType`, and Groq default-model constants added to `models.py`.
  - Provider selection wired through the `Reasoner` base class and `RoboticArm`.
  - Added example `examples/demo_groq_agent.py`, offline tests `tests/test_provider_groq.py`, and a "Using Groq" section below.

Gemini remains the default provider, so existing setups are unaffected.

## Get started

### Introduction

hCaptcha Challenger harnesses the spatial chain-of-thought (SCoT) reasoning capabilities of multimodal large language models (MLLMs) to construct an agentic workflow framework. This architecture empowers autonomous agents to perform zero-shot adaptation on diverse spatial-visual tasks through dynamic problem-solving workflows, eliminating the requirement for task-specific fine-tuning or additional training parameters.

The `Agent` controls browser pages through Playwright. In your workflow, the Agent is initialized with the `Page` object you pass in, allowing the Agent to take over interactions with the current page. You can implement two independent operations through the `Agent`: `click_checkbox` and `wait_for_challenge`.

hCaptcha is one of the pioneers in applying image diffusion and synthesis technology to the CAPTCHA domain. Benefiting from rapid advancements in automation engineering, hCaptcha can implement extremely frequent updates to its challenge types. Over the past two years, the community has increasingly struggled with handling such frequently changing human-machine challenges. Traditional convolutional neural networks (CNNs) face significant difficulties achieving good generalization on small datasets in object detection tasks. A comprehensive fine-tuning process typically requires substantial time and effort, often taking up to half a week to produce a CNN model suitable for production environments. However, by the time training is completed, hCaptcha might have already updated to new challenge types, rendering the recently trained model quickly obsolete or ineffective.

Therefore, the community urgently needs a robust, generalized visual solution capable of effectively tackling various spatial visual challenges. Regardless of how frequently hCaptcha updates its verification types, **this solution should swiftly adapt to environmental changes and autonomously control browsers to resolve various CAPTCHA tasks without human guidance.**

### Installation

```bash
uv pip install hcaptcha-challenger
```

### Quickstart

This document describes an automation approach for handling hCaptcha challenges, illustrating how to effectively interact with hCaptcha elements via the agent's robotic arm interface.

It is important to emphasize that the Agent interacts exclusively with web pages through the Page object. Consequently, the Agent can operate seamlessly on any platform or "patcher" built upon Playwright. In practical terms, this means that any browser supported and launched by Playwright can be utilized to execute the Agent using this approach.

In the following example, you need to create and set up your [GEMINI_API_KEY](https://aistudio.google.com/apikey):

```python
import asyncio
import json

from playwright.async_api import async_playwright, Page

from hcaptcha_challenger.agent import AgentV, AgentConfig
from hcaptcha_challenger.models import CaptchaResponse
from hcaptcha_challenger.utils import SiteKey


async def challenge(page: Page) -> AgentV:
    """Automates the process of solving an hCaptcha challenge."""
    # Initialize the agent configuration with API key (from parameters or environment)
    agent_config = AgentConfig()

    # Create an agent instance with the page and configuration
    # AgentV appears to be a specialized agent for visual challenges
    agent = AgentV(page=page, agent_config=agent_config)

    # Click the hCaptcha checkbox to initiate the challenge
    # The robotic_arm is an abstraction for performing UI interactions
    await agent.robotic_arm.click_checkbox()

    # Wait for the challenge to appear and be ready for solving
    # This may involve waiting for images to load or instructions to appear
    await agent.wait_for_challenge()

    # Note: The code ends here, suggesting this is part of a larger solution
    # that would continue with challenge solving steps after this point
    return agent


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()

        # Create a new page in the provided browser context
        page = await context.new_page()

        # Navigate to the hCaptcha test page using a predefined site key
        # SiteKey.user_easy likely refers to a test/demo hCaptcha with lower difficulty
        # await page.goto(SiteKey.as_site_link(SiteKey.discord))
        await page.goto(SiteKey.as_site_link(SiteKey.user_easy))

        # --- When you encounter hCaptcha in your workflow ---
        agent = await challenge(page)
        if agent.cr_list:
            cr: CaptchaResponse = agent.cr_list[-1]
            print(json.dumps(cr.model_dump(by_alias=True), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())

```

## Using Groq instead of Gemini

The agent supports [Groq](https://console.groq.com)'s vision models as an alternative backend. Create a `GROQ_API_KEY`, then set `LLM_PROVIDER="groq"`:

```python
agent_config = AgentConfig(
    LLM_PROVIDER="groq",
    # GROQ_API_KEY is read from the environment by default, or pass it explicitly:
    # GROQ_API_KEY="gsk_...",
)
```

When `LLM_PROVIDER="groq"` and the model fields are left at their defaults, they are switched automatically to Groq's vision models:

| Task | Default Groq model |
|------|--------------------|
| Challenge classifier | `meta-llama/llama-4-scout-17b-16e-instruct` |
| Image classifier / spatial reasoners | `meta-llama/llama-4-maverick-17b-128e-instruct` |

You can override any of them with the `CHALLENGE_CLASSIFIER_MODEL`, `IMAGE_CLASSIFIER_MODEL`, `SPATIAL_POINT_REASONER_MODEL`, and `SPATIAL_PATH_REASONER_MODEL` fields. Gemini remains the default provider, so existing setups are unaffected. See `examples/demo_groq_agent.py` for a complete runnable example.

> Note: Groq inlines images as base64 (max 4MB/image, 5 images per request) and uses best-effort JSON-schema structured output, falling back to JSON-object mode for models that don't support schema constraints.

## Dataset Collection

If you have your own solver, you can also use `hcaptcha-challenger` to manage image datasets:

```bash
uv venv
uv pip install -U hcaptcha-challenger
uv run hc dataset collect
```

![image_2025-04-12_18-33-07](assets/image_2025-04-12_18-33-07.png)

## Gallery

![image-20250402235820929](assets/image-20250402235820929.png)

### Image Label Binary

https://github.com/user-attachments/assets/c2cea4e0-82f4-466f-8c7a-20f8ea63732c

### Image Label Area Select

https://github.com/user-attachments/assets/42ce8b1d-bb17-4397-b7b0-a9f9578b740a

### Image Drag Drop

https://github.com/user-attachments/assets/c7720d20-ddb4-45e5-8008-e4c8f2de316d