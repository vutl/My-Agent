"""One tiny completion through the configured gateway; exit 0 when it answers.

    PYTHONPATH=backend backend/.venv/bin/python scripts/gateway_ping.py [model] [count]
"""

import sys

import httpx

from app.core.config import get_settings


def main() -> int:
    settings = get_settings()
    model = sys.argv[1] if len(sys.argv) > 1 else settings.default_model
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    ok = True
    for _ in range(count):
        response = httpx.post(
            settings.openai_api_base.rstrip("/") + "/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "Reply ok"}], "max_tokens": 5},
            headers={"Authorization": f"Bearer {settings.openai_api_key}"}, timeout=60,
        )
        print(model, response.status_code, response.text[:160])
        ok = ok and response.status_code == 200
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
