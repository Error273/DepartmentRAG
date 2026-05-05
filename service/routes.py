"""
Эндпоинты FastAPI сервиса.

/ask          — основной эндпоинт: вопрос → ответ + источники
/ask/stream   — SSE: источники + ответ через Server-Sent Events
/health       — проверка здоровья сервиса
"""

import json
import traceback

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from rag.pipeline import get_pipeline
from rag.config import LLM_MODEL
from service.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    SourceDocument,
)

router = APIRouter()


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Задать вопрос RAG-агенту",
    description="Принимает вопрос, агент самостоятельно ищет документы и генерирует ответ.",
)
async def ask(request: AskRequest):
    """Основной эндпоинт: вопрос → агент (поиск + LLM) → ответ."""
    try:
        pipeline = get_pipeline()
        response = pipeline.ask(
            question=request.question,
        )

        sources = [
            SourceDocument(
                title=doc.title,
                source_url=doc.source_url,
                score=round(doc.score, 4),
                match_type=doc.match_type,
            )
            for doc in response.sources
        ]

        return AskResponse(
            answer=response.answer,
            query=response.query,
            sources=sources,
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка генерации ответа: {e}")


@router.post(
    "/ask/stream",
    summary="Стриминговый ответ (SSE)",
    description="Источники и ответ приходят через Server-Sent Events.",
)
async def ask_stream(request: AskRequest):
    """SSE: запускает агента, отправляет источники и ответ как SSE-события."""

    def event_generator():
        try:
            pipeline = get_pipeline()
            response = pipeline.ask(question=request.question)

            # 1. Отправляем источники
            sources_data = [
                {
                    "title": doc.title,
                    "source_url": doc.source_url,
                    "score": round(doc.score, 4),
                    "match_type": doc.match_type,
                }
                for doc in response.sources
            ]
            yield f"event: sources\ndata: {json.dumps(sources_data, ensure_ascii=False)}\n\n"

            # 2. Отправляем ответ
            escaped = response.answer.replace("\n", "\\n")
            yield f"event: token\ndata: {escaped}\n\n"

            # 3. Сигнал завершения
            yield "event: done\ndata: [DONE]\n\n"

        except Exception as e:
            traceback.print_exc()
            error_msg = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"event: error\ndata: {error_msg}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── /health ──────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка здоровья сервиса",
)
async def health():
    """Проверяет, что сервис и Qdrant работают."""
    pipeline = get_pipeline()
    qdrant_ok = pipeline.check_qdrant()

    return HealthResponse(
        status="ok" if qdrant_ok else "degraded",
        model=LLM_MODEL,
        qdrant_connected=qdrant_ok,
    )
