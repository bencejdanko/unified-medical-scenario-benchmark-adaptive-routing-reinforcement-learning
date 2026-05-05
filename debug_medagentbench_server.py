"""Diagnostics for the MedAgentBench FHIR proxy service.

Uses MEDAGENTBENCH_API_KEY from the environment and never prints it.
"""

from __future__ import annotations

import argparse
import os
import time
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv

load_dotenv()


def headers() -> dict[str, str]:
    key = os.getenv("MEDAGENTBENCH_API_KEY")
    if not key:
        raise RuntimeError("MEDAGENTBENCH_API_KEY is required")
    return {"X-API-KEY": key}


def request(label: str, method: str, url: str, timeout: float) -> None:
    print(f"\n== {label} ==")
    print(f"{method} {url}")
    start = time.time()
    try:
        response = requests.request(method, url, headers=headers(), timeout=timeout)
        elapsed = time.time() - start
        print(f"status={response.status_code} elapsed={elapsed:.2f}s content_type={response.headers.get('content-type')}")
        text = response.text
        print(text[:1000] + ("..." if len(text) > 1000 else ""))
    except Exception as exc:
        elapsed = time.time() - start
        print(f"ERROR elapsed={elapsed:.2f}s {type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("MEDAGENTBENCH_BASE_URL", "http://localhost:8080"))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--reset-timeout", type=float, default=240.0)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    base = args.base_url.rstrip("/") + "/"
    request("health", "GET", urljoin(base, "health"), args.timeout)
    if args.reset:
        request("reset", "POST", urljoin(base, "reset"), args.reset_timeout)
    request("metadata", "GET", urljoin(base, "fhir/metadata"), args.timeout)
    request(
        "known patient",
        "GET",
        urljoin(base, "fhir/Patient?given=Peter&family=Stafford&birthdate=1932-12-29"),
        args.timeout,
    )


if __name__ == "__main__":
    main()
