# -*- coding: utf-8 -*-
"""
TempUploader - host local screenshots on a temporary file host so vision APIs
that only accept image *URLs* (e.g. aikit) can fetch them, then delete.

Configured from environment variables (shared by every URL-only provider):

- ``AIKIT_IMAGE_UPLOAD``   enable/disable uploading (default: enabled)
- ``AIKIT_UPLOAD_BASE_URL`` temp host base URL (default: https://tmp.malvryx.dev)
- ``AIKIT_UPLOAD_API_KEY``  optional ``X-API-Key`` for higher limits
- ``AIKIT_UPLOAD_EXPIRY``   temp file lifetime (default: ``1h``)

The names keep the ``AIKIT_`` prefix for backwards compatibility; they apply to
any provider that uploads images, not just aikit.
"""
import mimetypes
import os
from pathlib import Path

import httpx
from loguru import logger

# Default temporary file host used to give URL-only vision APIs a fetchable image.
DEFAULT_UPLOAD_BASE_URL = "https://tmp.malvryx.dev"


class TempUploader:
    """Upload/delete helper for the temporary image host."""

    def __init__(self):
        self.enabled = os.environ.get("AIKIT_IMAGE_UPLOAD", "true").lower() not in (
            "0",
            "false",
            "no",
        )
        self.base = os.environ.get("AIKIT_UPLOAD_BASE_URL", DEFAULT_UPLOAD_BASE_URL).rstrip("/")
        self.api_key = os.environ.get("AIKIT_UPLOAD_API_KEY", "")
        self.expiry = os.environ.get("AIKIT_UPLOAD_EXPIRY", "1h")

    async def upload(self, client: httpx.AsyncClient, path: Path) -> dict:
        """Upload one file; return ``{id, url, delete_token}``."""
        mime = mimetypes.guess_type(str(path))[0] or "image/png"
        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        with open(path, "rb") as fh:
            files = {"file": (path.name, fh, mime)}
            data = {"type": "temp", "expiry": self.expiry}
            resp = await client.post(
                f"{self.base}/upload", files=files, data=data, headers=headers
            )
        resp.raise_for_status()
        j = resp.json()
        url = j.get("cdnUrl") or j.get("directUrl")
        if not url:
            raise ValueError(f"Temp upload returned no URL: {j}")
        return {"id": j.get("id"), "url": url, "delete_token": j.get("deleteToken")}

    async def delete(self, client: httpx.AsyncClient, upload: dict) -> None:
        """Best-effort deletion of an uploaded temp file."""
        if not upload.get("id"):
            return
        try:
            headers = {}
            if upload.get("delete_token"):
                headers["X-Delete-Token"] = upload["delete_token"]
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            await client.request("DELETE", f"{self.base}/f/{upload['id']}", headers=headers)
        except Exception as e:  # pragma: no cover - best-effort cleanup
            logger.warning(f"Failed to delete temp upload {upload.get('id')}: {e}")
