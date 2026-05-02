"""
Скрипт 04: Эмбеддинги + загрузка в Qdrant + полные тексты документов.

1. Читает data/chunks/chunks.json → вычисляет эмбеддинги → загружает в Qdrant
2. Собирает полные тексты страниц из data/cleaned/ → сохраняет в doc_texts.json
3. Создаёт текстовый индекс в Qdrant для keyword-поиска

Запуск:
    python scripts/05_embed_and_index.py
"""

import json
import os
import sys
import time
from pathlib import Path

# Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    TextIndexParams,
    TokenizerType,
    VectorParams,
)

from rag.config import (
    COLLECTION_NAME,
    DOC_TEXTS_PATH,
    EMBEDDING_DIMENSION,
    QDRANT_HOST,
    QDRANT_PORT,
)
from rag.embedder import Embedder

# ── Пути ─────────────────────────────────────────────────────────────
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks.json"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"


def load_chunks(path: Path) -> list[dict]:
    """Загружает чанки из JSON-файла."""
    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Загружено чанков: {len(chunks)}")
    return chunks


def build_doc_texts(cleaned_dir: Path, output_path: Path) -> dict:
    """
    Собирает полные тексты документов из data/cleaned/.
    Ключ — source_url, значение — словарь с title, text, category.

    Сохраняет в doc_texts.json для использования ретривером.
    """
    doc_texts = {}

    for root, dirs, files in os.walk(cleaned_dir):
        for filename in sorted(files):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, cleaned_dir)

            # Определяем категорию
            if rel_path.startswith("news"):
                category = "news"
            elif rel_path.startswith("people"):
                category = "people"
            else:
                category = "main"

            with open(filepath, encoding="utf-8") as f:
                doc = json.load(f)

            url = doc.get("url", "")
            if not url:
                continue

            doc_texts[url] = {
                "title": doc.get("title", ""),
                "text": doc.get("content", "").strip(),
                "category": category,
            }

    # Сохраняем
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc_texts, f, ensure_ascii=False, indent=2)

    print(f"Собрано документов: {len(doc_texts)}")
    print(f"Сохранено в: {output_path}")
    return doc_texts


def create_collection(client: QdrantClient, name: str, dimension: int) -> None:
    """
    Создаёт коллекцию в Qdrant + текстовый индекс для keyword-поиска.
    """
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        print(f"Коллекция '{name}' уже существует — удаляю для пересоздания...")
        client.delete_collection(name)

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=dimension,
            distance=Distance.COSINE,
        ),
    )
    print(f"Коллекция '{name}' создана (dim={dimension}, metric=Cosine)")

    # Текстовый индекс для keyword-поиска по тексту чанка
    client.create_payload_index(
        collection_name=name,
        field_name="text",
        field_schema=TextIndexParams(
            type="text",
            tokenizer=TokenizerType.MULTILINGUAL,
            min_token_len=2,
            max_token_len=40,
        ),
    )
    print("Текстовый индекс на поле 'text' создан")

    # Текстовый индекс на title (для поиска по имени из заголовка)
    client.create_payload_index(
        collection_name=name,
        field_name="title",
        field_schema=TextIndexParams(
            type="text",
            tokenizer=TokenizerType.MULTILINGUAL,
            min_token_len=2,
            max_token_len=40,
        ),
    )
    print("Текстовый индекс на поле 'title' создан")


def upload_points(
    client: QdrantClient,
    collection_name: str,
    chunks: list[dict],
    embeddings,
    batch_size: int = 100,
) -> None:
    """Загружает точки (вектор + payload) в Qdrant батчами."""
    total = len(chunks)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        points = []

        for i in range(start, end):
            chunk = chunks[i]
            payload = {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "source_url": chunk["metadata"]["source_url"],
                "title": chunk["metadata"]["title"],
                "category": chunk["metadata"]["category"],
                "chunk_index": chunk["metadata"]["chunk_index"],
                "total_chunks": chunk["metadata"]["total_chunks"],
            }
            points.append(
                PointStruct(
                    id=i,
                    vector=embeddings[i].tolist(),
                    payload=payload,
                )
            )

        client.upsert(collection_name=collection_name, points=points)
        print(f"  Загружено {end}/{total} точек")


def main():
    # 1. Собираем полные тексты документов
    print("=" * 50)
    print("Шаг 1. Сборка полных текстов документов")
    print("=" * 50)
    build_doc_texts(CLEANED_DIR, DOC_TEXTS_PATH)

    # 2. Загрузка чанков
    print(f"\n{'=' * 50}")
    print("Шаг 2. Загрузка чанков и вычисление эмбеддингов")
    print("=" * 50)
    chunks = load_chunks(CHUNKS_PATH)
    texts = [chunk["text"] for chunk in chunks]

    # 3. Вычисление эмбеддингов
    embedder = Embedder()
    print(f"\nВычисляю эмбеддинги для {len(texts)} чанков...")
    t0 = time.time()
    embeddings = embedder.embed_batch(texts)
    elapsed = time.time() - t0
    print(f"Эмбеддинги вычислены за {elapsed:.1f} сек")
    print(f"Размерность: {embeddings.shape}")

    # 4. Подключение к Qdrant
    print(f"\n{'=' * 50}")
    print("Шаг 3. Загрузка в Qdrant")
    print("=" * 50)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print(f"Подключение к Qdrant ({QDRANT_HOST}:{QDRANT_PORT})...")
    collections = client.get_collections()
    print(f"Qdrant доступен. Коллекции: {[c.name for c in collections.collections]}")

    # 5. Создание коллекции + текстовые индексы
    create_collection(client, COLLECTION_NAME, EMBEDDING_DIMENSION)

    # 6. Загрузка точек
    print(f"\nЗагрузка {len(chunks)} точек...")
    upload_points(client, COLLECTION_NAME, chunks, embeddings)

    # 7. Проверка
    info = client.get_collection(COLLECTION_NAME)
    print(f"\n{'=' * 50}")
    print(f"Готово!")
    print(f"Коллекция: {COLLECTION_NAME}")
    print(f"Точек: {info.points_count}")
    print(f"Размерность: {info.config.params.vectors.size}")
    print(f"Метрика: {info.config.params.vectors.distance}")
    print(f"doc_texts.json: {DOC_TEXTS_PATH}")
    print("=" * 50)

    # 8. Smoke-test
    test_query = "Поташев кабинет"
    print(f"\n[Smoke-test] semantic: '{test_query}'")
    query_vector = embedder.embed(test_query)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=3,
    )
    for i, point in enumerate(results.points, 1):
        print(f"  #{i} (score={point.score:.4f}) {point.payload['title']}")

    # Keyword test
    from qdrant_client.models import FieldCondition, Filter, MatchText

    print(f"\n[Smoke-test] keyword: 'Поташев'")
    scroll_results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="text",
                    match=MatchText(text="Поташев"),
                )
            ],
        ),
        limit=5,
    )
    for point in scroll_results[0]:
        print(f"  id={point.id} | {point.payload['title']} | {point.payload['text'][:80]}...")


if __name__ == "__main__":
    main()
