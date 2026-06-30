# -*- coding: utf-8 -*-
"""
Live debug stream of the solver browser.

When ``HCAPTCHA_API_STREAM_ENABLED`` is set, the API runs a second, lightweight
HTTP server on ``STREAM_PORT`` that serves a **continuous MJPEG stream**. The
stream is up from the moment the server starts (showing a "No Task" placeholder)
and switches to the live browser the instant a solve begins — the stream
connection itself never breaks, so you can join from VLC (or a phone browser over
Tailscale) *before* sending a solve request and watch the whole thing happen.

How it stays continuous:

- A single shared :class:`StreamHub` holds the *latest* JPEG frame.
- The MJPEG generator emits that latest frame at a fixed FPS, forever, regardless
  of whether a solve is running. When idle it re-emits the placeholder; during a
  solve the solver feeds it real frames via a Chrome DevTools screencast.

Because there is only one frame buffer, this mode supports exactly one concurrent
solve (the API forces ``MAX_CONCURRENT_SOLVES=1`` when streaming is on).
"""
import asyncio
import io
from typing import AsyncIterator

from loguru import logger
from PIL import Image, ImageDraw, ImageFont

_BG = (15, 17, 21)
_FG = (235, 237, 240)
_ACCENT = (88, 166, 255)
_MUTED = (140, 146, 158)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=size)
    except Exception:  # pragma: no cover - very old Pillow
        return ImageFont.load_default()


class StreamHub:
    """Shared single-frame buffer feeding the MJPEG stream."""

    def __init__(
        self,
        *,
        fps: int = 6,
        max_width: int = 1280,
        quality: int = 60,
        idle_text: str = "No Task",
    ):
        self.fps = max(1, fps)
        self.max_width = max_width
        self.max_height = int(max_width * 9 / 16)
        self.quality = quality
        self._idle_text = idle_text
        self._status = idle_text
        self._active = False
        self._frame_count = 0
        self._idle_frame = self._render(idle_text, "Waiting for a solve request")
        self._latest = self._idle_frame

    # -- producers -----------------------------------------------------------
    def push(self, jpeg: bytes) -> None:
        """Set the latest frame from a raw JPEG (from the browser screencast)."""
        if jpeg:
            self._latest = jpeg
            self._frame_count += 1

    def set_active(self, label: str) -> None:
        """Mark a solve as started; show a banner until live frames arrive."""
        self._active = True
        self._frame_count = 0
        self._status = label
        self._latest = self._render(label, "Connecting to the browser…")

    def set_idle(self) -> None:
        """Return to the idle placeholder once a solve finishes."""
        self._active = False
        self._status = self._idle_text
        self._latest = self._idle_frame

    # -- consumers -----------------------------------------------------------
    @property
    def snapshot(self) -> bytes:
        return self._latest

    @property
    def status(self) -> str:
        return self._status

    async def mjpeg(self) -> AsyncIterator[bytes]:
        """Yield multipart MJPEG chunks of the latest frame at a fixed FPS."""
        interval = 1.0 / self.fps
        while True:
            frame = self._latest
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
            )
            await asyncio.sleep(interval)

    # -- rendering -----------------------------------------------------------
    def _render(self, title: str, subtitle: str = "") -> bytes:
        w, h = self.max_width, self.max_height
        img = Image.new("RGB", (w, h), _BG)
        draw = ImageDraw.Draw(img)

        title_font = _load_font(max(28, w // 22))
        sub_font = _load_font(max(16, w // 48))

        def _centered(text: str, font, y: int, fill):
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            draw.text(((w - tw) / 2, y), text, font=font, fill=fill)

        _centered(title, title_font, int(h * 0.40), _FG if title == self._idle_text else _ACCENT)
        if subtitle:
            _centered(subtitle, sub_font, int(h * 0.40) + max(40, w // 18), _MUTED)
        _centered("hCaptcha Solver — debug stream", sub_font, int(h * 0.85), _MUTED)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return buf.getvalue()


def create_stream_app(hub: StreamHub):
    """Build a tiny FastAPI app that serves the MJPEG stream and a viewer page."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, Response, StreamingResponse

    app = FastAPI(title="hCaptcha Solver — Debug Stream", docs_url=None, redoc_url=None)

    _VIEWER = """<!DOCTYPE html><html><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Solver debug stream</title>
<style>body{margin:0;background:#0f1115;color:#ebedf0;font-family:system-ui,sans-serif;
text-align:center}img{max-width:100%;height:auto}h3{font-weight:500;color:#8c929e}</style>
</head><body><h3>hCaptcha Solver — live debug stream</h3>
<img src="/stream.mjpeg" alt="stream"/>
<p style="color:#8c929e">VLC: Media → Open Network Stream → this URL + <code>/stream.mjpeg</code></p>
</body></html>"""

    @app.get("/", response_class=HTMLResponse)
    async def viewer():
        return _VIEWER

    @app.get("/stream.mjpeg")
    async def stream():
        return StreamingResponse(
            hub.mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.get("/snapshot.jpg")
    async def snapshot():
        return Response(content=hub.snapshot, media_type="image/jpeg")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "active": hub.status}

    return app


class ScreencastSession:
    """
    Attach a Chrome DevTools screencast to a Playwright page and pump JPEG frames
    into a :class:`StreamHub`. Use as an async context manager around a solve.
    """

    def __init__(self, hub: StreamHub):
        self._hub = hub
        self._client = None

    async def start(self, ctx, page) -> None:
        try:
            self._client = await ctx.new_cdp_session(page)
            loop = asyncio.get_event_loop()

            def _on_frame(params):
                import base64

                try:
                    self._hub.push(base64.b64decode(params["data"]))
                finally:
                    # Frames stop arriving unless each is acked.
                    sid = params.get("sessionId")
                    if sid is not None and self._client is not None:
                        loop.create_task(
                            self._client.send(
                                "Page.screencastFrameAck", {"sessionId": sid}
                            )
                        )

            self._client.on("Page.screencastFrame", _on_frame)
            await self._client.send(
                "Page.startScreencast",
                {
                    "format": "jpeg",
                    "quality": self._hub.quality,
                    "maxWidth": self._hub.max_width,
                    "maxHeight": self._hub.max_height,
                    "everyNthFrame": 1,
                },
            )
        except Exception as e:  # pragma: no cover - debug feature, never fail a solve
            logger.warning(f"Failed to start debug screencast: {e}")
            self._client = None

    async def stop(self) -> None:
        if self._client is None:
            return
        try:
            await self._client.send("Page.stopScreencast")
        except Exception:  # pragma: no cover
            pass
        try:
            await self._client.detach()
        except Exception:  # pragma: no cover
            pass
        self._client = None
