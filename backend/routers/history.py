"""
History Router
==============
GET  /api/history        — Retrieve scan history (last 100 entries).
DELETE /api/history      — Clear all history.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse

from models.schemas import HistoryEntry, HistoryResponse

router = APIRouter(prefix="/api", tags=["History"])


def get_history(request: Request) -> list:
    return request.app.state.history


# ── GET /api/history ──────────────────────────────────────────────────────────

@router.get(
    "/history",
    response_model=HistoryResponse,
    summary="Retrieve crop scan history",
    description=(
        "Returns the last 100 scan records stored in-memory. "
        "Each record includes the timestamp, detected class, confidence, and severity."
    ),
)
async def get_history_route(history: list = Depends(get_history)):
    recent = list(reversed(history[-100:]))  # newest first
    entries = [HistoryEntry(**h) for h in recent]
    return HistoryResponse(success=True, count=len(entries), data=entries)


# ── DELETE /api/history ───────────────────────────────────────────────────────

@router.delete(
    "/history",
    summary="Clear all scan history",
)
async def clear_history(history: list = Depends(get_history)):
    history.clear()
    return JSONResponse(content={"success": True, "message": "History cleared."})
