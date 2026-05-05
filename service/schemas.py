"""
Pydantic-модели запросов и ответов для FastAPI сервиса.
"""

from pydantic import BaseModel, Field


# ── Запросы ──────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    """Запрос к RAG-системе."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Вопрос пользователя о кафедре аэрогидромеханики",
        examples=["Кто заведует кафедрой?"],
    )



# ── Ответы ───────────────────────────────────────────────────────────

class SourceDocument(BaseModel):
    """Один источник, использованный для ответа."""
    title: str
    source_url: str
    score: float
    match_type: str  # "semantic", "bm25", "hybrid"


class AskResponse(BaseModel):
    """Ответ RAG-системы."""
    answer: str = Field(description="Ответ LLM на вопрос")
    query: str = Field(description="Исходный вопрос")
    sources: list[SourceDocument] = Field(
        description="Список источников, использованных для ответа"
    )


class HealthResponse(BaseModel):
    """Ответ health-эндпоинта."""
    status: str = "ok"
    model: str = Field(description="Используемая LLM-модель")
    qdrant_connected: bool = Field(description="Подключение к Qdrant")


class ErrorResponse(BaseModel):
    """Ответ при ошибке."""
    detail: str
