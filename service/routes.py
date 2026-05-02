"""
Эндпоинты FastAPI сервиса.

/ask          — основной эндпоинт: вопрос → ответ + источники
/ask/stream   — SSE-стриминг: токены приходят по мере генерации
/health       — проверка здоровья сервиса
"""

import json
import traceback

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from rag.pipeline import RAGPipeline
from rag.config import LLM_MODEL
from service.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    SourceDocument,
)

router = APIRouter()

# Pipeline инициализируется один раз при первом импорте роутов
# (загрузка моделей, подключение к Qdrant)
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline



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
    description="Токены ответа приходят по мере генерации через Server-Sent Events.",
)
async def ask_stream(request: AskRequest):
    """SSE-стриминг: сначала отправляет источники, потом токены ответа."""

    def event_generator():
        try:
            pipeline = get_pipeline()

            # 1. Поиск документов
            docs = pipeline.retriever.search(
                query=request.question,
                top_k=request.top_k,
            )

            # 2. Отправляем источники первым SSE-событием
            sources_data = [
                {
                    "title": doc.title,
                    "source_url": doc.source_url,
                    "score": round(doc.score, 4),
                    "match_type": doc.match_type,
                }
                for doc in docs
            ]
            yield f"event: sources\ndata: {json.dumps(sources_data, ensure_ascii=False)}\n\n"

            # 3. Формируем контекст и стримим ответ
            context = pipeline.retriever.format_context(docs)
            for token in pipeline.llm.ask_stream(
                question=request.question,
                context=context,
            ):
                # Экранируем переносы строк для SSE
                escaped = token.replace("\n", "\\n")
                yield f"event: token\ndata: {escaped}\n\n"

            # 4. Сигнал завершения
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
    qdrant_ok = False
    try:
        pipeline = get_pipeline()
        # Простая проверка подключения к Qdrant
        pipeline.retriever.client.get_collections()
        qdrant_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok" if qdrant_ok else "degraded",
        model=LLM_MODEL,
        qdrant_connected=qdrant_ok,
    )
