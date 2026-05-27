"""Run intent-style searches against a running semcode API."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from typing import Any

DEFAULT_QUERIES = [
    "validate JWT token",
    "extract user id from token",
    "hash password plaintext",
    "format date as ISO string",
    "build URL with query parameters",
]


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000", help="semcode API base URL")
    parser.add_argument("--k", type=int, default=3, help="Results per query")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds")
    parser.add_argument("queries", nargs="*", default=DEFAULT_QUERIES)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    health = _get_json(f"{base_url}/health", args.timeout)
    print(f"health: status={health.get('status')} ready={health.get('ready')}")

    for query in args.queries:
        params = urllib.parse.urlencode({"q": query, "k": args.k})
        url = f"{base_url}/search?{params}"
        print(f"\nquery: {query}")
        try:
            body = _get_json(url, args.timeout)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8")
            print(f"  error: HTTP {exc.code} {error_body}")
            continue

        for result in body.get("results", []):
            print(
                "  #{rank} {symbol_name} ({file_path}:{start_line}) score={score}".format(**result)
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
