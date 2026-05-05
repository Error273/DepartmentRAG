"""
FastAPI приложение — ML-сервис для RAG-системы кафедры аэрогидромеханики.

Запуск:
    uvicorn service.app:app --reload --host 0.0.0.0 --port 8000

Документация:
    http://localhost:8000/docs     — Swagger UI
    http://localhost:8000/redoc    — ReDoc
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from service.routes import router
from rag.pipeline import get_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Инициализация при старте:
    загружает модели, подключается к Qdrant, прогревает pipeline.
    """
    print("🚀 Запуск сервиса...")
    get_pipeline()  # Предварительная инициализация
    print("✅ Сервис готов к работе!")
    yield
    print("🛑 Сервис остановлен.")


app = FastAPI(
    title="RAG-сервис кафедры аэрогидромеханики КФУ",
    description=(
        "Интеллектуальный помощник кафедры аэрогидромеханики.\n\n"
        "Принимает вопросы на русском языке и отвечает, "
        "используя информацию с сайта кафедры."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — разрешаем всё для разработки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем роуты
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service.app:app", host="0.0.0.0", port=8000, reload=True)
