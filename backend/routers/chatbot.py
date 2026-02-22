"""
Chatbot Router
==============
POST /api/chat  — PlantCare AI agricultural chatbot powered by Groq.
GET  /api/chat/history  — Retrieve conversation history.
DEL  /api/chat/history  — Clear conversation history.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request

from models.schemas import ChatMessage, ChatRequest, ChatResponse, ChatHistoryResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Chatbot"])

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

SYSTEM_PROMPT = """You are PlantCare AI Assistant, an expert agricultural chatbot specializing in:
- Plant disease identification and treatment
- Crop management and best practices
- Fertilizer and NPK recommendations
- Pesticide guidance (chemical and organic)
- Disease prevention strategies
- Soil health and irrigation advice

You have knowledge of 38 plant disease categories covering: Apple, Blueberry, Cherry, Corn/Maize, 
Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.

Always give practical, actionable advice. When recommending pesticides, include dosage and safety info.
If a farmer describes symptoms, help identify the likely disease and suggest treatment steps.
Keep responses concise but complete. Use simple language suitable for farmers.
If asked something outside agriculture/plant care, politely redirect to your area of expertise."""


def get_groq_client():
    """Initialise Groq client — raises clearly if API key is missing."""
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=503,
            detail=(
                "GROQ_API_KEY environment variable is not set. "
                "Get a free key at https://console.groq.com and run: "
                "export GROQ_API_KEY=your_key_here"
            ),
        )
    try:
        from groq import Groq
        return Groq(api_key=GROQ_API_KEY)
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="groq package is not installed. Run: pip install groq",
        )


def get_chat_history(request: Request) -> list:
    if not hasattr(request.app.state, "chat_history"):
        request.app.state.chat_history = []
    return request.app.state.chat_history


# ── POST /api/chat ────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with the PlantCare AI agricultural assistant",
    description=(
        "Send a message to the Groq-powered PlantCare AI chatbot. "
        "The bot specialises in plant diseases, crop management, and fertilizer advice. "
        "Conversation history is maintained in-memory for the session."
    ),
)
async def chat(payload: ChatRequest, request: Request):
    client       = get_groq_client()
    chat_history = get_chat_history(request)

    # Keep last 20 messages for context (10 turns)
    recent_history = chat_history[-20:] if len(chat_history) > 20 else chat_history

    # Build messages list for Groq
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for entry in recent_history:
        messages.append({"role": entry["role"], "content": entry["content"]})
    messages.append({"role": "user", "content": payload.message})

    # Call Groq API
    try:
        completion = client.chat.completions.create(
            model=payload.model or "llama3-8b-8192",
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        reply = completion.choices[0].message.content
    except Exception as exc:
        logger.exception("Groq API error: %s", exc)
        raise HTTPException(status_code=502, detail=f"Groq API error: {exc}")

    # Persist to history
    timestamp = datetime.now(timezone.utc).isoformat()
    chat_history.append({"role": "user",      "content": payload.message, "timestamp": timestamp})
    chat_history.append({"role": "assistant", "content": reply,           "timestamp": timestamp})

    # Cap history at 200 messages
    if len(chat_history) > 200:
        del chat_history[0:2]

    logger.info("Chat | model=%s | user=%s chars | reply=%s chars",
                payload.model or "llama3-8b-8192", len(payload.message), len(reply))

    return ChatResponse(
        success=True,
        message=reply,
        model=payload.model or "llama3-8b-8192",
        tokens_used=completion.usage.total_tokens if completion.usage else None,
    )


# ── GET /api/chat/history ─────────────────────────────────────────────────────

@router.get(
    "/chat/history",
    response_model=ChatHistoryResponse,
    summary="Get conversation history",
)
async def get_history(request: Request):
    history = get_chat_history(request)
    entries = [ChatMessage(**h) for h in history]
    return ChatHistoryResponse(success=True, count=len(entries), data=entries)


# ── DELETE /api/chat/history ──────────────────────────────────────────────────

@router.delete("/chat/history", summary="Clear conversation history")
async def clear_history(request: Request):
    get_chat_history(request).clear()
    return {"success": True, "message": "Chat history cleared."}
