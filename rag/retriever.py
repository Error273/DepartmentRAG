"""
Retriever: гибридный поиск (semantic + BM25) с возвратом релевантных чанков.

Стратегия:
  1. Semantic search — cosine similarity через Qdrant (смысловой поиск)
  2. BM25 — классический лексический поиск с TF-IDF (точные слова)
  3. Normalized Score Fusion — объединение результатов через нормализованные скоры
  4. Chunk window expansion — для каждого найденного чанка подтягиваем ±1 соседний
     чтобы не терять контекст на стыке чанков
"""

import json
import math
import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient

from rag.config import (
    CHUNKS_PATH,
    COLLECTION_NAME,
    DOC_TEXTS_PATH,
    QDRANT_HOST,
    QDRANT_PORT,
    SEMANTIC_TOP_K,
    TOP_K,
)
from rag.embedder import Embedder


@dataclass
class RetrievedDocument:
    """Результат поиска — целая страница с метаданными."""
    source_url: str
    title: str
    category: str
    full_text: str
    score: float  # гибридный скор (0..1)
    bm25_norm: float = 0.0  # нормализованный BM25 скор
    sem_norm: float = 0.0  # нормализованный semantic скор
    match_type: str = ""  # "semantic", "bm25", "hybrid"


class Retriever:
    """
    Гибридный ретривер: BM25 + semantic поиск, объединённые через RRF.

    BM25 автоматически обрабатывает IDF: слово «кафедра», которое есть везде,
    получает низкий вес. А «Поташев» или «Гульнара» — высокий.
    Никаких стоп-слов и ручной настройки скоров.

    Использование:
        retriever = Retriever()
        results = retriever.search("в каком кабинете Поташев?")
        for doc in results:
            print(doc.title, doc.rrf_score)
            print(doc.full_text)
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        client: QdrantClient | None = None,
        collection_name: str = COLLECTION_NAME,
        doc_texts_path: str | None = None,
    ):
        self.embedder = embedder or Embedder()
        self.client = client or QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.collection_name = collection_name

        # Загрузка полных текстов документов
        path = doc_texts_path or str(DOC_TEXTS_PATH)
        with open(path, encoding="utf-8") as f:
            self.doc_texts: dict = json.load(f)
        print(f"Загружено {len(self.doc_texts)} полных документов")

        # Загрузка чанков для window expansion (url → chunk_index → text)
        self.chunk_lookup: dict[str, dict[int, str]] = {}
        with open(str(CHUNKS_PATH), encoding="utf-8") as f:
            all_chunks = json.load(f)
        for chunk in all_chunks:
            url = chunk["metadata"]["source_url"]
            idx = chunk["metadata"]["chunk_index"]
            if url not in self.chunk_lookup:
                self.chunk_lookup[url] = {}
            self.chunk_lookup[url][idx] = chunk["text"]
        print(f"Загружено {len(all_chunks)} чанков для window expansion")

        # Строим BM25-индекс по ПОЛНЫМ документам (не чанкам).
        # Так BM25-ранг будет на уровне страниц — совпадает с гранулярностью возврата.
        self.doc_urls: list[str] = []  # URL в том же порядке что и BM25-корпус
        corpus_tokens: list[list[str]] = []

        for url, doc in self.doc_texts.items():
            # Токенизируем заголовок + текст для лучшего матча
            combined = doc.get("title", "") + " " + doc.get("text", "")
            corpus_tokens.append(self._tokenize(combined))
            self.doc_urls.append(url)

        self.bm25 = BM25Okapi(corpus_tokens)
        print(f"BM25-индекс построен: {len(self.doc_urls)} документов")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Простая токенизация для BM25: lowercase + разбивка по словам.
        Без стоп-слов — BM25 сам обработает частотные слова через IDF.
        """
        text = text.lower()
        tokens = re.findall(r"[а-яёa-z0-9]+", text)
        return [t for t in tokens if len(t) >= 2]

    def _get_full_doc(self, source_url: str) -> dict | None:
        """Получает полный текст документа по URL."""
        return self.doc_texts.get(source_url)

    def semantic_search(
        self,
        query: str,
        top_k: int = SEMANTIC_TOP_K,
    ) -> list[dict]:
        """Семантический поиск через Qdrant. Возвращает чанки с текстом."""
        query_vector = self.embedder.embed(query).tolist()

        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
            )
        except Exception as e:
            print(f"[WARNING] Qdrant недоступен, используем только BM25: {e}")
            return []

        return [
            {
                "source_url": p.payload.get("source_url", ""),
                "title": p.payload.get("title", ""),
                "category": p.payload.get("category", ""),
                "chunk_text": p.payload.get("text", ""),
                "chunk_index": p.payload.get("chunk_index", 0),
                "score": p.score,
            }
            for p in results.points
        ]

    def bm25_search(
        self,
        query: str,
        top_k: int = SEMANTIC_TOP_K,
    ) -> list[dict]:
        """BM25 лексический поиск по полным документам."""
        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1]

        hits = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            if len(hits) >= top_k:
                break

            url = self.doc_urls[idx]
            doc = self.doc_texts[url]
            cat = doc.get("category", "")

            hits.append({
                "source_url": url,
                "title": doc.get("title", ""),
                "category": cat,
                "score": float(scores[idx]),
            })

        return hits

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        alpha: float = 0.5,
    ) -> list[RetrievedDocument]:
        """
        Гибридный поиск: semantic + BM25, объединённые через нормализованные скоры.

        hybrid_score(doc) = alpha * norm_bm25(doc) + (1 - alpha) * norm_semantic(doc)

        Скоры нормализуются min-max в [0, 1], поэтому разница в BM25 скорах
        (напр. 19.3 vs 2.6 для «Гульнара Талгатовна») сохраняется.

        Args:
            query: Текстовый запрос.
            top_k: Сколько уникальных страниц вернуть.
            alpha: Вес BM25 (0..1). 0.5 = равный вес.

        Returns:
            Список RetrievedDocument с полными текстами страниц.
        """
        # 1. Два параллельных поиска
        semantic_hits = self.semantic_search(query)
        bm25_hits = self.bm25_search(query)

        # 2. Собираем скоры по URL (дедупликация semantic — берём лучший скор)
        semantic_scores: dict[str, float] = {}
        for hit in semantic_hits:
            url = hit["source_url"]
            if url:
                semantic_scores[url] = max(semantic_scores.get(url, 0), hit["score"])

        bm25_scores: dict[str, float] = {}
        for hit in bm25_hits:
            url = hit["source_url"]
            if url:
                bm25_scores[url] = max(bm25_scores.get(url, 0), hit["score"])

        # 2b. Собираем релевантные чанки по URL из semantic-поиска
        #     Для каждого документа сохраняем найденные чанки (отсортированные по chunk_index)
        url_chunks: dict[str, list[dict]] = {}
        for hit in semantic_hits:
            url = hit["source_url"]
            if url and "chunk_text" in hit and hit["chunk_text"]:
                if url not in url_chunks:
                    url_chunks[url] = []
                # Дедупликация чанков по chunk_index
                existing_indices = {c["chunk_index"] for c in url_chunks[url]}
                if hit.get("chunk_index", 0) not in existing_indices:
                    url_chunks[url].append({
                        "chunk_text": hit["chunk_text"],
                        "chunk_index": hit.get("chunk_index", 0),
                    })

        # 3. Нормализация скоров в [0, 1]
        #    Semantic: уже cosine similarity в [0, 1] — используем как есть
        #    BM25: используем сигмоиду — sigmoid(x) = 1 / (1 + e^(-k*(x - x0)))
        #    Это сохраняет абсолютное значение скора (19.3 >> 2.6).
        def sigmoid_norm(scores: dict[str, float], k: float = 1.0, x0: float = 3.0) -> dict[str, float]:
            return {url: 1.0 / (1.0 + math.exp(-k * (v - x0))) for url, v in scores.items()}

        norm_sem = dict(semantic_scores)  # уже в [0, 1]
        norm_bm25 = sigmoid_norm(bm25_scores)

        # 4. Собираем метаданные
        url_meta: dict[str, dict] = {}
        for hit in semantic_hits + bm25_hits:
            url = hit["source_url"]
            if url and url not in url_meta:
                url_meta[url] = {
                    "title": hit["title"],
                    "category": hit["category"],
                }

        # 5. Считаем гибридный скор
        #    hybrid = max(bm25, sem) + alpha * min(bm25, sem)
        #    Так один сильный сигнал (BM25=1.0 для «Гульнара») побеждает
        #    два средних (bm25=0.4 + sem=0.6).
        #    Бонус alpha за наличие второго сигнала.
        all_urls = set(norm_sem.keys()) | set(norm_bm25.keys())
        hybrid_results = []

        for url in all_urls:
            s_bm25 = norm_bm25.get(url, 0.0)
            s_sem = norm_sem.get(url, 0.0)
            hybrid_score = max(s_bm25, s_sem) + alpha * min(s_bm25, s_sem)

            # Определяем тип матча
            has_bm25 = url in bm25_scores
            has_sem = url in semantic_scores
            if has_bm25 and has_sem:
                match_type = "hybrid"
            elif has_bm25:
                match_type = "bm25"
            else:
                match_type = "semantic"

            hybrid_results.append({
                "url": url,
                "hybrid_score": hybrid_score,
                "bm25_norm": s_bm25,
                "sem_norm": s_sem,
                "match_type": match_type,
                **url_meta.get(url, {}),
            })

        # 6. Сортируем по гибридному скору
        hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # 7. Берём top_k и формируем результаты с релевантными чанками + window expansion
        results = []
        for item in hybrid_results[:top_k]:
            url = item["url"]

            # Собираем текст из релевантных чанков с расширением окна ±1
            if url in url_chunks and url_chunks[url]:
                # Индексы найденных чанков
                found_indices = {c["chunk_index"] for c in url_chunks[url]}

                # Расширяем окно: для каждого найденного чанка добавляем ±1 сосед
                expanded_indices = set()
                for idx in found_indices:
                    expanded_indices.add(idx - 1)
                    expanded_indices.add(idx)
                    expanded_indices.add(idx + 1)

                # Собираем текст из chunk_lookup (все чанки этого документа)
                doc_chunks = self.chunk_lookup.get(url, {})
                final_indices = sorted(idx for idx in expanded_indices if idx in doc_chunks)
                context_text = "\n\n".join(doc_chunks[idx] for idx in final_indices)
            else:
                # Fallback: BM25-only документ — берём полный текст
                doc = self._get_full_doc(url)
                context_text = doc["text"] if doc else "(текст недоступен)"

            results.append(
                RetrievedDocument(
                    source_url=url,
                    title=item.get("title", ""),
                    category=item.get("category", ""),
                    full_text=context_text,
                    score=item["hybrid_score"],
                    bm25_norm=item["bm25_norm"],
                    sem_norm=item["sem_norm"],
                    match_type=item["match_type"],
                )
            )

        return results

    def format_context(self, docs: list[RetrievedDocument]) -> str:
        """
        Форматирует найденные документы в контекст для LLM.
        Отправляет полные тексты страниц.
        """
        if not docs:
            return "Контекст не найден."

        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(
                f"[Источник {i}] {doc.title}\n"
                f"URL: {doc.source_url}\n\n"
                f"{doc.full_text}"
            )
        return "\n\n" + ("=" * 40 + "\n\n").join(parts)
