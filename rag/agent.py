"""
RAG-агент на LangGraph: LLM самостоятельно решает когда и что искать.

Архитектура:
  1. LLM получает вопрос пользователя
  2. Формулирует поисковый запрос и вызывает tool search_documents
  3. Анализирует результаты — если недостаточно, переформулирует и ищет снова
  4. После max_iterations неудачных попыток — отвечает что не нашёл
  5. Если нашёл — формирует ответ с источниками
"""

import time
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError

from rag.config import (
    YANDEX_CLOUD_API_KEY,
    YANDEX_CLOUD_FOLDER,
    BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    SYSTEM_PROMPT,
    AGENT_MAX_ITERATIONS,
)
from rag.retriever import Retriever, RetrievedDocument


# ── Глобальный ретривер (инициализируется один раз) ──────────────────

_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    """Lazy-init ретривера (тяжёлая операция: загрузка моделей + BM25)."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


# ── Tool: поиск документов ───────────────────────────────────────────

@tool
def search_documents(query: str) -> str:
    """Поиск информации по базе данных кафедры аэрогидромеханики КФУ.

    Используй этот инструмент, чтобы найти документы по запросу.
    Формулируй запрос кратко — ключевые слова и имена работают лучше длинных фраз.

    Примеры хороших запросов:
    - "заведующий кафедрой" — найти кто заведует
    - "Поташев расписание кабинет" — найти информацию о конкретном преподавателе
    - "научные направления исследования" — найти научные направления
    - "дисциплины бакалавриат" — найти список дисциплин

    Args:
        query: Поисковый запрос. Ключевые слова и имена собственные дают лучшие результаты.
    """
    retriever = get_retriever()
    docs = retriever.search(query=query, top_k=3)

    if not docs:
        return "По данному запросу ничего не найдено. Попробуй другие ключевые слова."

    # Форматируем результаты для LLM
    # Ретривер теперь возвращает релевантные чанки, а не полные документы,
    # поэтому текст уже сфокусирован и не требует обрезки.
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(
            f"[Документ {i}] {doc.title}\n"
            f"URL: {doc.source_url}\n"
            f"Релевантность: {doc.score:.2f}\n\n"
            f"{doc.full_text}"
        )

    return "\n\n" + ("=" * 40 + "\n\n").join(parts)


# ── Dataclass для логов и ответа агента ──────────────────────────────

@dataclass
class ToolCallLog:
    """Лог одного вызова инструмента."""
    tool_name: str
    arguments: dict[str, Any]
    result: str


@dataclass
class AgentResponse:
    """Ответ RAG-агента."""
    answer: str
    sources: list[RetrievedDocument] = field(default_factory=list)
    query: str = ""
    search_queries: list[str] = field(default_factory=list)  # какие запросы агент делал
    tool_logs: list[ToolCallLog] = field(default_factory=list)  # логи вызовов инструментов
    elapsed_seconds: float = 0.0  # время ответа в секундах
    total_tokens: int = 0  # общее количество токенов (input + output)
    blocked: bool = False  # заблокировано guardrail-ом


# ── RAG Agent ────────────────────────────────────────────────────────

class RAGAgent:
    """
    RAG-агент на базе LangGraph ReAct.

    LLM самостоятельно решает:
    - Когда вызывать поиск
    - Какой запрос сформулировать
    - Нужно ли искать повторно с другим запросом
    - Когда достаточно информации для ответа

    Использование:
        agent = RAGAgent()
        response = agent.ask("Кто заведует кафедрой?")
        print(response.answer)
    """

    def __init__(self):
        print("🤖 Инициализация RAG Agent...")

        # Инициализируем ретривер (если ещё не инициализирован)
        get_retriever()

        # LLM через Yandex AI Studio (OpenAI-совместимый API)
        self.llm = ChatOpenAI(
            model=f"gpt://{YANDEX_CLOUD_FOLDER}/{LLM_MODEL}",
            api_key="unused",
            openai_api_base=BASE_URL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            default_headers={
                "Authorization": f"Api-Key {YANDEX_CLOUD_API_KEY}",
            },
            model_kwargs={"reasoning_effort": "none"},
        )

        # Создаём ReAct-агента с инструментом поиска
        self.tools = [search_documents]
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=SYSTEM_PROMPT,
        )

        # recursion_limit: каждая итерация агента = 2 шага (LLM call + tool call)
        # 3 итерации поиска + финальный ответ = 3*2 + 1 = 7,
        # берём с большим запасом для моделей с длинными рассуждениями
        self.recursion_limit = AGENT_MAX_ITERATIONS * 2 + 10

        print("✅ RAG Agent готов!")

    def ask(
        self,
        question: str,
        history: list[dict] | None = None,
    ) -> AgentResponse:
        """
        Задать вопрос агенту.

        Args:
            question: Вопрос пользователя.
            history: Опциональная история диалога
                     [{\"role\": \"user\"/\"assistant\", \"content\": \"...\"}].

        Returns:
            AgentResponse с ответом и метаданными.
        """
        # Формируем сообщения
        messages = []

        # Добавляем историю
        if history:
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

        # Добавляем текущий вопрос
        messages.append(HumanMessage(content=question))

        try:
            # Запускаем агента с замером времени
            start_time = time.time()
            result = self.agent.invoke(
                {"messages": messages},
                config={"recursion_limit": self.recursion_limit},
            )
            elapsed = time.time() - start_time

            # Извлекаем ответ — последнее AI-сообщение
            response_messages = result["messages"]
            answer = ""
            for msg in reversed(response_messages):
                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    answer = msg.content
                    break

            # LangGraph может вставить английское сообщение при достижении recursion_limit
            # вместо исключения GraphRecursionError
            if not answer or "need more steps" in answer.lower():
                answer = (
                    "К сожалению, не удалось сформировать ответ. "
                    "Попробуйте переформулировать вопрос."
                )

            # Собираем запросы и логи инструментов
            search_queries = []
            tool_logs = []

            # Собираем tool_call_id → ToolCallLog для сопоставления с результатами
            pending_calls: dict[str, ToolCallLog] = {}

            for msg in response_messages:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        log_entry = ToolCallLog(
                            tool_name=tc["name"],
                            arguments=tc["args"],
                            result="",
                        )
                        tool_logs.append(log_entry)
                        # Сохраняем для последующего сопоставления с ToolMessage
                        if "id" in tc:
                            pending_calls[tc["id"]] = log_entry

                        if tc["name"] == "search_documents":
                            q = tc["args"].get("query", "")
                            if q:
                                search_queries.append(q)

                elif isinstance(msg, ToolMessage):
                    # Сопоставляем результат с вызовом инструмента
                    tc_id = getattr(msg, "tool_call_id", None)
                    if tc_id and tc_id in pending_calls:
                        # Обрезаем результат для логов (первые 500 символов)
                        content = msg.content if isinstance(msg.content, str) else str(msg.content)
                        pending_calls[tc_id].result = content[:500]
                        if len(content) > 500:
                            pending_calls[tc_id].result += "... [обрезано]"

            # Получаем реальные источники из последнего поиска
            sources = self._extract_sources(search_queries)

            # Считаем токены из usage_metadata всех AI-сообщений
            total_tokens = 0
            for msg in response_messages:
                if isinstance(msg, AIMessage):
                    usage = getattr(msg, "usage_metadata", None)
                    if usage and isinstance(usage, dict):
                        total_tokens += usage.get("total_tokens", 0)

            return AgentResponse(
                answer=answer,
                sources=sources,
                query=question,
                search_queries=search_queries,
                tool_logs=tool_logs,
                elapsed_seconds=round(elapsed, 2),
                total_tokens=total_tokens,
            )

        except GraphRecursionError:
            # Агент превысил лимит итераций — не нашёл ответ
            return AgentResponse(
                answer=(
                    "К сожалению, после нескольких попыток поиска "
                    "мне не удалось найти релевантную информацию по вашему вопросу "
                    "в базе данных кафедры.\n\n"
                    "Попробуйте:\n"
                    "• Переформулировать вопрос\n"
                    "• Использовать другие ключевые слова\n"
                    "• Уточнить, что именно вас интересует"
                ),
                sources=[],
                query=question,
                search_queries=[],
            )

        except Exception as e:
            print(f"[RAGAgent] Ошибка: {e}")
            raise

    def _extract_sources(
        self, search_queries: list[str]
    ) -> list[RetrievedDocument]:
        """
        Получает источники из последнего поискового запроса агента.
        Показываем только результаты последнего успешного поиска.
        """
        if not search_queries:
            return []

        retriever = get_retriever()
        last_query = search_queries[-1]
        return retriever.search(query=last_query, top_k=3)
