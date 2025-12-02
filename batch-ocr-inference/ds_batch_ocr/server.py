from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import threading
import time
from typing import Any, Awaitable, Dict, List, Optional, Sequence

import requests
from openai import AsyncOpenAI

from .document import encode_image

LOGGER = logging.getLogger(__name__)


def _stream_output(pipe, prefix: str) -> None:
    try:
        for line in iter(pipe.readline, ""):
            print(f"[{prefix}] {line.rstrip()}", flush=True)
    finally:
        pipe.close()


def launch_vllm() -> subprocess.Popen:
    model_id = os.environ.get("MODEL_ID", "deepseek-ai/DeepSeek-OCR")
    served_name = os.environ.get("SERVED_MODEL_NAME", "deepseek-ocr")
    port = os.environ.get("PORT", "8080")
    host = os.environ.get("HOST", "0.0.0.0")

    cmd: List[str] = [
        "vllm",
        "serve",
        "--model",
        model_id,
        "--served-model-name",
        served_name,
        "--tensor-parallel-size",
        os.environ.get("TENSOR_PARALLEL_SIZE", "1"),
        "--max-model-len",
        os.environ.get("MAX_MODEL_LEN", "4096"),
        "--gpu-memory-utilization",
        os.environ.get("GPU_MEMORY_UTILIZATION", "0.85"),
        "--port",
        port,
        "--host",
        host,
        "--trust-remote-code",
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb",
        os.environ.get("MM_PROCESSOR_CACHE_GB", "0"),
        "--logits-processors",
        os.environ.get(
            "LOGITS_PROCESSORS",
            "vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor",
        ),
    ]

    extra_server_args = os.environ.get("EXTRA_VLLM_ARGS")
    if extra_server_args:
        cmd.extend(extra_server_args.split())

    LOGGER.info("Launching vLLM server with command: %s", " ".join(cmd))
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    threads = []
    for name, pipe in (("STDOUT", process.stdout), ("STDERR", process.stderr)):
        if pipe is not None:
            thread = threading.Thread(
                target=_stream_output,
                args=(pipe, f"vLLM {name}"),
                daemon=True,
            )
            thread.start()
            threads.append(thread)

    process._log_threads = threads  # type: ignore[attr-defined]
    return process


def shutdown_server(server_process: subprocess.Popen) -> None:
    LOGGER.info("Shutting down vLLM server")
    server_process.send_signal(signal.SIGTERM)
    try:
        server_process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        LOGGER.warning("Server did not exit in time, sending SIGKILL")
        server_process.kill()

    log_threads = getattr(server_process, "_log_threads", [])
    for thread in log_threads:
        thread.join(timeout=1)


def wait_for_server(url: str, timeout_s: int = 300, interval_s: int = 5) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=5)
            if response.ok:
                return True
        except Exception:
            pass
        time.sleep(interval_s)
    return False


def should_launch_server() -> bool:
    return os.environ.get("SKIP_SERVER_LAUNCH", "").lower() not in {"1", "true", "yes"}


def base_url_from_env() -> str:
    port = os.environ.get("PORT", "8080")
    default_url = f"http://127.0.0.1:{port}"
    return os.environ.get("BASE_URL", default_url)


def prepare_payload(
    image: "Image.Image",
    served_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    return {
        "model": served_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_image(image)}"},
                    },
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "extra_body": {
            "skip_special_tokens": False,
            "vllm_xargs": {
                "ngram_size": 30,
                "window_size": 90,
                "whitelist_token_ids": "[128821,128822]",
            },
        },
    }


class DeepSeekClient:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        max_tokens: int,
        temperature: float,
        *,
        request_timeout: int = 120,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
        max_retry_wait_seconds: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature
        self.default_request_timeout = request_timeout
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self.max_retry_wait_seconds = max_retry_wait_seconds

        client_base = f"{self.base_url.rstrip('/')}/v1"
        self._client = AsyncOpenAI(api_key="vllm", base_url=client_base)

    async def _async_completion(
        self,
        payload: Dict[str, Any],
        request_timeout: int,
    ) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=payload["model"],
                messages=payload["messages"],
                max_tokens=payload["max_tokens"],
                temperature=payload["temperature"],
                timeout=request_timeout,
                extra_body=payload.get("extra_body"),
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.error("DeepSeek request failed: %s", exc)
            raise

        if not response.choices:
            return ""
        message = response.choices[0].message
        return getattr(message, "content", "") or ""

    def infer(self, requests_data: Sequence[Dict[str, Any]]) -> List[str]:
        if not requests_data:
            return []

        payloads = []
        timeouts = []
        for req in requests_data:
            payloads.append(
                prepare_payload(
                    image=req["image"],
                    served_name=self.model_name,
                    prompt=req.get("prompt", ""),
                    max_tokens=req.get("max_tokens", self.default_max_tokens),
                    temperature=req.get("temperature", self.default_temperature),
                )
            )
            timeouts.append(req.get("request_timeout") or self.default_request_timeout)

        return self._run_async(self._async_infer_batch(payloads, timeouts))

    async def _async_infer_batch(
        self,
        payloads: Sequence[Dict[str, Any]],
        timeouts: Sequence[int],
    ) -> List[str]:
        tasks = [
            asyncio.create_task(self._async_completion(payload, timeout))
            for payload, timeout in zip(payloads, timeouts)
        ]
        return await asyncio.gather(*tasks)

    def close(self) -> None:
        try:
            self._run_async(self._client.aclose())
        except AttributeError:
            pass

    @staticmethod
    def _run_async(coro: Awaitable[Any]) -> Any:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(coro)
            loop.run_until_complete(loop.shutdown_asyncgens())
            return result
        finally:
            asyncio.set_event_loop(None)
            loop.close()


__all__ = [
    "launch_vllm",
    "shutdown_server",
    "wait_for_server",
    "should_launch_server",
    "base_url_from_env",
    "DeepSeekClient",
]


