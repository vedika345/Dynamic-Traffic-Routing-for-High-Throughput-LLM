"""
ModelPool
=========

Owns all communication with Ollama and implements the
"load balancing + failover" part of the router:

- load balancing: if you list multiple `base_urls` in
  config/router_config.yaml (e.g. several Ollama instances on
  different ports/machines), requests are spread across them
  round-robin style.

- failover: each call to `generate()` is given a *chain* of models
  to try -- the model picked by the router first, then the global
  `fallback_chain` from the config. If a model fails (connection
  error, timeout, model not pulled, etc.) the pool automatically
  tries the next model in the chain on the next base_url.
"""

import time
import logging
import requests

logger = logging.getLogger("model_pool")
logging.basicConfig(level=logging.INFO)


class ModelPool:
    def __init__(self, config: dict):
        self.base_urls = config["ollama"]["base_urls"]
        self.models = config["models"]
        self.fallback_chain = config.get("fallback_chain", [])
        self._rr_index = 0

    def _next_base_url(self) -> str:
        """Round-robin selection across configured Ollama instances."""
        url = self.base_urls[self._rr_index % len(self.base_urls)]
        self._rr_index += 1
        return url

    def _call_ollama(self, base_url: str, model_name: str, prompt: str, timeout: int) -> str:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")

    def generate(self, model_key: str, prompt: str, timeout: int = 120) -> dict:
        """
        Try the model mapped to `model_key`, falling back through
        `fallback_chain` if it fails. Returns a dict describing what
        actually happened (useful for logging/debugging).
        """
        primary_model = self.models.get(model_key)

        # Build the ordered list of *actual* Ollama model tags to try,
        # without duplicates.
        chain = [primary_model] + list(self.fallback_chain)
        seen = set()
        chain = [m for m in chain if m and not (m in seen or seen.add(m))]

        last_error = None
        for attempt, model_name in enumerate(chain):
            base_url = self._next_base_url()
            try:
                start = time.time()
                text = self._call_ollama(base_url, model_name, prompt, timeout=timeout)
                latency = time.time() - start
                return {
                    "response": text,
                    "model_used": model_name,
                    "base_url": base_url,
                    "latency": latency,
                    "fallback_used": attempt > 0,
                    "attempt": attempt,
                }
            except Exception as e:
                logger.warning("Model '%s' at %s failed: %s", model_name, base_url, e)
                last_error = e
                continue

        raise RuntimeError(
            f"All models in fallback chain {chain} failed. Last error: {last_error}"
        )
