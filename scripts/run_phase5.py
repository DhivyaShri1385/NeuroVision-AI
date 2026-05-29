"""
Phase 5 — Start the FastAPI server.

Usage:
    python scripts/run_phase5.py               # default: host=0.0.0.0, port=8000
    python scripts/run_phase5.py --port 9000
    python scripts/run_phase5.py --reload      # hot-reload (dev mode)

Then open:
    http://localhost:8000/docs   ← Swagger UI
    http://localhost:8000/redoc  ← ReDoc
    http://localhost:8000/health ← health check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeuroVision AI — Phase 5 API server")
    p.add_argument("--host",   default="0.0.0.0", help="Bind host (default 0.0.0.0)")
    p.add_argument("--port",   type=int, default=8000, help="Bind port (default 8000)")
    p.add_argument("--reload", action="store_true",
                   help="Enable auto-reload on code changes (development mode).")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of worker processes (use 1 when --reload).")
    p.add_argument("--log-level", default="info",
                   choices=["debug", "info", "warning", "error"],
                   help="Uvicorn log level.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    import uvicorn
    print()
    print("=" * 56)
    print("  NeuroVision AI — FastAPI Backend")
    print("=" * 56)
    print(f"  URL   : http://{args.host}:{args.port}")
    print(f"  Docs  : http://localhost:{args.port}/docs")
    print(f"  Health: http://localhost:{args.port}/health")
    print(f"  Reload: {args.reload}")
    print("=" * 56)
    print()

    uvicorn.run(
        "app.main:app",
        host      = args.host,
        port      = args.port,
        reload    = args.reload,
        workers   = 1 if args.reload else args.workers,
        log_level = args.log_level,
    )


if __name__ == "__main__":
    main()
