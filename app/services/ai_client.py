"""
Dual AI Provider Client for SmartTraffic AI.
Automatically tries providers in order:
  1. OLLAMA (primary, free, local) — fast, no API cost
  2. SUMOPOD (fallback, API key required) — reliable cloud

Usage:
    from app.services.ai_client import call_ai, call_ai_vision

    # Smart auto-select (Ollama → SumoPod fallback)
    result = call_ai("What is the traffic on Jl Sudirman?")
    print(result["content"], result["provider"], result["model"])

    # Vision (with image)
    result = call_ai_vision("Read the plate", image_np=frame)
    print(result["content"])
"""
import json
import os
import time
import urllib.request
import urllib.error
import base64
import threading
import cv2
import numpy as np

# ── Auto-load .env ───────────────────────────────────────────────────────────
# ai_client.py works standalone (not just via run.py)
try:
    _cwd_env = os.path.join(os.getcwd(), ".env")
    if os.path.exists(_cwd_env):
        try:
            from dotenv import load_dotenv
            load_dotenv(_cwd_env, override=False)
        except Exception:
            pass
except Exception:
    pass

# ── Ollama helpers ──────────────────────────────────────────────────────────────

def _ollama_url():
    return (os.environ.get("OLLAMA_URL") or os.environ.get("AI_BASE_URL") or "http://localhost:11434").strip().rstrip("/")

_ollama_model = None
_ollama_model_at = 0.0
_ollama_models = None
_ollama_models_at = 0.0
_ollama_reachable_cache = {}
_ollama_reachable_at = 0.0
_llock = threading.Lock()

def _resolve_ollama_model():
    global _ollama_model, _ollama_model_at
    now = time.time()
    if _ollama_model and (now - _ollama_model_at) < 300:
        return _ollama_model
    env = os.environ.get("OLLAMA_MODEL") or os.environ.get("AI_MODEL") or ""
    if env.strip():
        _ollama_model = env.strip()
        _ollama_model_at = now
        return _ollama_model
    models = _fetch_ollama_models()
    if models:
        _ollama_model = models[0]
        _ollama_model_at = now
        return _ollama_model
    return None

def _sanitize(name):
    return str(name or "").strip().lower().replace(":", "-")

def _fetch_ollama_models(force=False):
    global _ollama_models, _ollama_models_at
    now = time.time()
    if not force and _ollama_models and (now - _ollama_models_at) < 120:
        return _ollama_models
    try:
        req = urllib.request.Request(
            f"{_ollama_url()}/api/tags",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=4) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        parsed = json.loads(raw) if raw else {}
        items = parsed.get("models") or []
        names = [_sanitize(m.get("name") or "") for m in items]
        _ollama_models = [n for n in names if n]
        _ollama_models_at = now
        return _ollama_models
    except Exception:
        return _ollama_models or []

def _ollama_reachable(timeout=3):
    global _ollama_reachable_cache, _ollama_reachable_at
    now = time.time()
    if now - _ollama_reachable_at < 30:
        return _ollama_reachable_cache.get("ok", False)
    try:
        req = urllib.request.Request(
            f"{_ollama_url()}/api/tags",
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=timeout)
        result = True
    except Exception:
        result = False
    with _llock:
        _ollama_reachable_cache = {"ok": result}
        _ollama_reachable_at = now
    return result

# ── API config ─────────────────────────────────────────────────────────────────

def _api_config():
    return {
        "provider": os.environ.get("AI_PROVIDER") or "sumopod",
        "base_url": os.environ.get("AI_BASE_URL") or "https://ai.sumopod.com/v1",
        "api_key": os.environ.get("AI_API_KEY") or "",
        "model": os.environ.get("AI_CHAT_MODEL") or os.environ.get("AI_MODEL") or "gpt-4o-mini",
    }

def _api_available():
    cfg = _api_config()
    return bool(cfg["api_key"] and cfg["base_url"])

# ── Core call ──────────────────────────────────────────────────────────────────

def call_ai(
    message,
    system_prompt=None,
    provider=None,
    timeout=None,
    max_tokens=1200,
    temperature=0.2,
) -> dict:
    """Call AI with dual-provider fallback.
    
    Strategy:
      1. Auto/ollama: Try Ollama first (15s) → SumoPod fallback (30s)
      2. api/sumopod: Use SumoPod directly
      3. Returns {"content", "provider", "model", "success", "error", "latency_ms"}
    """
    start = time.time()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": message})

    auto_mode = provider is None
    use_api_only = provider in ("sumopod", "openai", "api", "gemini", "deepseek", "groq")

    # Try Ollama first
    if (auto_mode or provider == "ollama") and not use_api_only:
        if _ollama_reachable(timeout=3):
            res = _call_ollama(messages, timeout=timeout or 15, max_tokens=max_tokens, temperature=temperature)
            if res["success"]:
                res["latency_ms"] = int((time.time() - start) * 1000)
                return res
            print(f"[AI-CLIENT] Ollama failed: {res.get('error')}, trying API fallback...")

    # Try API provider (SumoPod)
    if _api_available():
        res = _call_api(messages, timeout=timeout or 30, max_tokens=max_tokens, temperature=temperature)
        if res["success"]:
            res["latency_ms"] = int((time.time() - start) * 1000)
            return res
        print(f"[AI-CLIENT] API failed: {res.get('error')}")

    # Last try: Ollama if not yet attempted
    if not _ollama_reachable(timeout=3):
        return {
            "content": "",
            "provider": "error",
            "model": "",
            "success": False,
            "error": "Tidak ada AI provider tersedia. Pastikan Ollama running atau AI_API_KEY di-set di .env",
            "latency_ms": int((time.time() - start) * 1000),
        }
    res = _call_ollama(messages, timeout=60, max_tokens=max_tokens, temperature=temperature)
    res["latency_ms"] = int((time.time() - start) * 1000)
    return res

def _call_ollama(messages, timeout=15, max_tokens=1200, temperature=0.2):
    model = _resolve_ollama_model()
    if not model:
        return {"success": False, "error": "No Ollama model resolved"}

    body = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": messages,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": 4096,
            "top_k": 20,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        },
    }

    for attempt in range(2):
        try:
            req = urllib.request.Request(
                f"{_ollama_url()}/api/chat",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                parsed = json.loads(resp.read().decode("utf-8", errors="ignore") or "{}")
            if not isinstance(parsed, dict):
                return {"success": False, "error": "Non-JSON response"}
            if parsed.get("error"):
                err = str(parsed.get("error"))
                if attempt == 0 and "model" in err.lower() and "not found" in err.lower():
                    models = _fetch_ollama_models(force=True)
                    if models and models[0] != model:
                        body["model"] = models[0]
                        global _ollama_model, _ollama_model_at
                        _ollama_model = models[0]
                        _ollama_model_at = time.time()
                        continue
                return {"success": False, "error": err}
            content = str(parsed.get("message", {}).get("content") or parsed.get("response") or "").strip()
            if not content:
                return {"success": False, "error": "Empty response"}
            return {"content": content, "provider": "ollama", "model": model, "success": True, "error": None}
        except urllib.error.HTTPError as e:
            try:
                raw = e.read().decode("utf-8", errors="ignore")
            except Exception:
                raw = ""
            return {"success": False, "error": f"HTTP {e.code}: {raw}"}
        except Exception as e:
            if attempt == 0:
                continue
            return {"success": False, "error": str(e)}
    return {"success": False, "error": "Max retries"}

def _call_api(messages, timeout=30, max_tokens=1200, temperature=0.2):
    cfg = _api_config()
    if not cfg["api_key"]:
        return {"success": False, "error": "No API key"}

    try:
        req = urllib.request.Request(
            f"{cfg['base_url'].rstrip('/')}/chat/completions",
            data=json.dumps({
                "model": cfg["model"],
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {cfg['api_key']}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            parsed = json.loads(resp.read().decode("utf-8"))
        content = str(parsed.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        if not content:
            return {"success": False, "error": "Empty response"}
        return {"content": content, "provider": cfg["provider"], "model": cfg["model"], "success": True, "error": None}
    except urllib.error.HTTPError as e:
        try:
            raw = e.read().decode("utf-8", errors="ignore")
            edata = json.loads(raw) if raw else {}
            emsg = edata.get("error", {}).get("message") or str(e)
        except Exception:
            emsg = str(e)
        return {"success": False, "error": f"HTTP {e.code}: {emsg}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ── Vision ─────────────────────────────────────────────────────────────────────

def call_ai_vision(message, image_base64=None, image_np=None, system_prompt=None,
                    provider=None, timeout=30) -> dict:
    """Call AI with image input (vision-capable providers only)."""
    img_b64 = image_base64
    if image_np is not None and img_b64 is None:
        try:
            _, buf = cv2.imencode(".jpg", image_np, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        except Exception:
            img_b64 = None

    if not img_b64:
        return call_ai(message, system_prompt=system_prompt, provider=provider, timeout=timeout)

    # Vision only works with API providers (SumoPod/OpenAI)
    res = _call_api_vision(message, img_b64, timeout=timeout, system_prompt=system_prompt)
    if not res["success"]:
        return call_ai(message, system_prompt=system_prompt, provider=provider, timeout=timeout)
    return res

def _call_api_vision(message, img_b64, timeout=30, system_prompt=None):
    cfg = _api_config()
    if not cfg["api_key"]:
        return {"success": False, "error": "No API key"}

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": message},
        ],
    })

    try:
        req = urllib.request.Request(
            f"{cfg['base_url'].rstrip('/')}/chat/completions",
            data=json.dumps({
                "model": cfg["model"],
                "messages": messages,
                "max_tokens": 2000,
                "temperature": 0.1,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {cfg['api_key']}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            parsed = json.loads(resp.read().decode("utf-8"))
        content = str(parsed.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        if not content:
            return {"success": False, "error": "Empty vision response"}
        return {"content": content, "provider": cfg["provider"], "model": cfg["model"], "success": True, "error": None}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ── Status & test ─────────────────────────────────────────────────────────────

def get_ai_status() -> dict:
    """Return status of all AI providers."""
    ollama_ok = _ollama_reachable(timeout=3)
    api_ok = _api_available()
    return {
        "ollama": {
            "available": ollama_ok,
            "model": _resolve_ollama_model() if ollama_ok else None,
            "url": _ollama_url(),
        },
        "api": {
            "available": api_ok,
            "provider": _api_config()["provider"],
            "model": _api_config()["model"],
            "base_url": _api_config()["base_url"],
        },
        "strategy": "ollama_primary_api_fallback",
        "primary": "ollama" if ollama_ok else ("api" if api_ok else "none"),
    }

def ai_test_connection() -> dict:
    """Test AI connection."""
    status = get_ai_status()
    if status["primary"] != "none":
        res = call_ai("OK", timeout=20, max_tokens=10)
        return {
            "status": "success" if res["success"] else "error",
            "connected": res["success"],
            "provider": res.get("provider", "unknown"),
            "model": res.get("model", ""),
            "latency_ms": res.get("latency_ms", 0),
            "error": res.get("error"),
            "status_detail": status,
        }
    return {
        "status": "error",
        "connected": False,
        "message": "Tidak ada AI provider tersedia",
        "status_detail": status,
    }
