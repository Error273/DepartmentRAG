"""
Embedder: обёртка над sentence-transformers для получения эмбеддингов текста.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

from rag.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION


class Embedder:
    """
    Обёртка над моделью эмбеддингов.

    Использование:
        embedder = Embedder()
        vector = embedder.embed("Какой-то текст")           # один вектор
        vectors = embedder.embed_batch(["текст1", "текст2"]) # батч
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        print(f"Загрузка модели: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        self.dimension = EMBEDDING_DIMENSION
        print(f"Модель загружена. Размерность вектора: {self.dimension}")

    def embed(self, text: str) -> np.ndarray:
        """
        Получить эмбеддинг одного текста.

        Args:
            text: Входной текст.

        Returns:
            numpy-массив размером (dimension,).
        """
        return self.model.encode(text, normalize_embeddings=True)

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Получить эмбеддинги для списка текстов.

        Args:
            texts: Список строк.
            batch_size: Размер батча (для CPU 64 — хороший баланс скорости и памяти).
            show_progress: Показывать прогресс-бар.

        Returns:
            numpy-массив размером (len(texts), dimension).
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
