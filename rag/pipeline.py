"""
Pipeline: обратно-совместимая обёртка над RAG-агентом.

Сохраняет тот же интерфейс (ask/ask_stream), но делегирует
всю работу в RAGAgent (LangGraph ReAct).
"""

from dataclasses import dataclass

from rag.agent import RAGAgent, AgentResponse, ToolCallLog
from rag.retriever import RetrievedDocument


@dataclass
class RAGResponse:
    """Полный ответ RAG-системы (обратная совместимость)."""
    answer: str                          # Ответ LLM
    sources: list[RetrievedDocument]     # Найденные документы
    query: str                           # Исходный вопрос
    tool_logs: list[ToolCallLog] = None  # Логи вызовов инструментов
    elapsed_seconds: float = 0.0         # Время ответа в секундах
    total_tokens: int = 0                # Общее количество токенов


class RAGPipeline:
    """
    Полный RAG-пайплайн через LangGraph-агента.

    Обратно-совместимая обёртка: тот же интерфейс .ask(),
    но внутри LLM сама решает когда и что искать.

    Использование:
        pipeline = RAGPipeline()
        response = pipeline.ask("Кто заведует кафедрой?")
        print(response.answer)
        for src in response.sources:
            print(f"  - {src.title}: {src.source_url}")
    """

    def __init__(self):
        print("Инициализация RAG Pipeline (агентный режим)...")
        self.agent = RAGAgent()
        print("RAG Pipeline готов к работе!")

    def ask(
        self,
        question: str,
        top_k: int = 5,
        history: list[dict] | None = None,
    ) -> RAGResponse:
        """
        Полный цикл RAG через агента.

        Args:
            question: Вопрос пользователя.
            top_k: Не используется напрямую (агент сам управляет поиском).
            history: История диалога [{\"role\": ..., \"content\": ...}].

        Returns:
            RAGResponse с ответом, источниками и исходным запросом.
        """
        agent_response: AgentResponse = self.agent.ask(
            question=question,
            history=history,
        )

        return RAGResponse(
            answer=agent_response.answer,
            sources=agent_response.sources,
            query=agent_response.query,
            tool_logs=agent_response.tool_logs,
            elapsed_seconds=agent_response.elapsed_seconds,
            total_tokens=agent_response.total_tokens,
        )
