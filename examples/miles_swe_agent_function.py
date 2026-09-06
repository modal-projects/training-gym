"""Harbor agent client for the Miles SWE-agent example.

This module is the value passed to `--custom-agent-function-path` in Miles. It is
called once per rollout sample and forwards the request to an external Harbor
agent server, returning the reward and agent metrics.
"""

import asyncio
import logging
import os
import socket
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx

logger = logging.getLogger(__name__)

_agent_server_client: httpx.AsyncClient | None = None


def _get_agent_server_client() -> httpx.AsyncClient:
    """Return a long-lived client with TCP keepalive for idle network paths."""
    global _agent_server_client
    if _agent_server_client is None:
        socket_options = [
            (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
            (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPIDLE", 4), 60),
            (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPINTVL", 5), 30),
            (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPCNT", 6), 5),
        ]
        transport = httpx.AsyncHTTPTransport(socket_options=socket_options)
        _agent_server_client = httpx.AsyncClient(
            transport=transport,
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
            timeout=None,
        )
    return _agent_server_client


def _rewrite_host_port(original: str, external_host: str) -> str:
    """Replace the host in a ``host:port`` string while preserving the port."""
    if ":" in original:
        host, _, port_str = original.rpartition(":")
        try:
            port = int(port_str)
            return f"{external_host}:{port}"
        except ValueError:
            return external_host
    return external_host


async def _post_agent_server(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    client = _get_agent_server_client()
    response = await client.post(url, json=payload)
    response.raise_for_status()
    return response.json()


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    """Run a single task instance through the Harbor agent server."""
    metadata = metadata or {}
    request_kwargs = request_kwargs or {}

    agent_server_url = os.getenv(
        "AGENT_SERVER_URL",
        os.getenv("SWE_AGENT_URL", "http://localhost:11000"),
    )
    model_name = os.getenv(
        "AGENT_MODEL_NAME",
        os.getenv("SWE_AGENT_MODEL_NAME", "model"),
    )

    session_url = f"{base_url}/v1"
    external_host = os.getenv("MILES_ROUTER_EXTERNAL_HOST")
    if external_host:
        parsed = urlparse(session_url)
        port = parsed.port
        netloc = f"{external_host}:{port}" if port else external_host
        session_url = urlunparse(parsed._replace(netloc=netloc))

    request: dict[str, Any] = {
        **metadata,
        "base_url": session_url,
        "model": f"openai/{model_name}",
        "sampling_params": request_kwargs,
    }

    max_seq_len = metadata.get("max_seq_len")
    if max_seq_len is not None:
        request["max_seq_len"] = int(max_seq_len)

    session_server_id = metadata.get("session_server_id")
    if session_server_id is not None and external_host:
        session_server_id = _rewrite_host_port(session_server_id, external_host)
        request["session_server_id"] = session_server_id

    session_server_instance_id = metadata.get("session_server_instance_id")
    if session_server_instance_id is not None:
        request["session_server_instance_id"] = session_server_instance_id

    try:
        response = await asyncio.wait_for(
            _post_agent_server(f"{agent_server_url}/run", request),
            timeout=3600,
        )
    except asyncio.TimeoutError:
        logger.error("Agent server call timed out after 3600s")
        return None
    except asyncio.CancelledError:
        logger.warning("Agent server call cancelled")
        raise
    except Exception as e:
        logger.error(f"Agent server call failed: {e}")
        return None

    return {
        "reward": response.get("reward", 0.0),
        "exit_status": response.get("exit_status", ""),
        "eval_report": response.get("eval_report", {}),
        "agent_metrics": response.get("agent_metrics", {}),
    }


async def abort(args) -> None:
    """Best-effort flush of in-flight agent tasks on rollout abort."""
    agent_server_url = os.getenv("AGENT_SERVER_URL", os.getenv("SWE_AGENT_URL"))
    instance_id = getattr(args, "session_server_instance_id", None)
    if not agent_server_url or not instance_id:
        return

    headers = None
    admin_secret = os.getenv("HARBOR_ADMIN_SECRET")
    if admin_secret:
        headers = {"Authorization": f"Bearer {admin_secret}"}

    try:
        client = _get_agent_server_client()
        result = await client.post(
            f"{agent_server_url.rstrip('/')}/flush",
            json={"session_server_instance_id": instance_id},
            headers=headers,
        )
        result.raise_for_status()
        logger.info(f"Flushed agent server {agent_server_url}: {result.json()}")
    except Exception as e:
        logger.warning(f"Failed to flush agent server {agent_server_url}: {e}")
