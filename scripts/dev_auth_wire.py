#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import urllib.request
from urllib.error import HTTPError, URLError


def post_json(url: str, payload: dict) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, body
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return e.code, body
    except URLError as e:
        return 0, str(e)


def extract_token(body: str) -> str | None:
    try:
        obj = json.loads(body)
    except Exception:
        return None
    tok = obj.get("access_token") or obj.get("token")
    return tok if isinstance(tok, str) and tok.strip() else None


def main() -> int:
    base = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
    username = os.environ.get("USERNAME", "admin")
    password = os.environ.get("PASSWORD", "admin")

    print(f"[wire] BASE_URL={base}")
    print(f"[wire] USERNAME={username}")

    login_url = f"{base}/api/v1/auth/login"
    bootstrap_url = f"{base}/api/v1/auth/bootstrap-admin"

    status, body = post_json(login_url, {"username": username, "password": password})
    tok = extract_token(body)

    if not tok:
        print("[wire] Login failed or no token returned. Attempting to create default admin...")
        post_json(bootstrap_url, {"username": username, "password": password})
        status, body = post_json(login_url, {"username": username, "password": password})
        tok = extract_token(body)

    if not tok:
        print("[wire] Could not obtain a token.")
        print(f"[wire] Last status={status}")
        print(body)
        return 1

    print("\n[wire] ✅ JWT obtained.\n")
    print("Option A (recommended): open this URL once (it will self-scrub the token from the URL):")
    print(f"  {base}/?jwt={tok}")
    print("\nOption B: paste this in the browser console:")
    print(f"  localStorage.setItem('mongars_jwt', '{tok}'); location.reload();")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
