"""
ASHA Sahayak — Python Client
HTTP client for interacting with the environment server.
"""

from __future__ import annotations

import httpx
from typing import Any, Dict, List, Optional


class AshaClient:
    """
    Synchronous HTTP client for the ASHA Sahayak environment.

    Usage:
        client = AshaClient(base_url="http://localhost:7860")
        obs = client.reset(task_id="easy", seed=42)
        obs = client.step(question="Does the child have fever?")
        obs = client.step(
            referral_decision="REFER_IMMEDIATELY",
            urgency="immediate",
            primary_concern="severe_pneumonia",
        )
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def reset(self, task_id: str = "easy", seed: int = 42) -> Dict[str, Any]:
        resp = self._client.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id, "seed": seed},
        )
        resp.raise_for_status()
        return resp.json()["observation"]

    def step(
        self,
        referral_decision: str = "PENDING",
        urgency: str = "monitor",
        primary_concern: str = "",
        action_items: Optional[List[str]] = None,
        question: Optional[str] = None,
        confidence: float = 0.8,
    ) -> Dict[str, Any]:
        resp = self._client.post(
            f"{self.base_url}/step",
            json={
                "referral_decision": referral_decision,
                "urgency": urgency,
                "primary_concern": primary_concern,
                "action_items": action_items or [],
                "question": question,
                "confidence": confidence,
            },
        )
        resp.raise_for_status()
        return resp.json()["observation"]

    def state(self) -> Dict[str, Any]:
        resp = self._client.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()["state"]

    def health(self) -> str:
        resp = self._client.get(f"{self.base_url}/health")
        resp.raise_for_status()
        data = resp.json()
        return data.get("status", "healthy")

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
