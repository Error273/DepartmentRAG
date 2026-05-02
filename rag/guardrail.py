"""
Guardrail: защита от prompt injection, jailbreaking и off-topic запросов.

Быстрая и дешёвая модель (gpt-oss-20b) классифицирует входящее сообщение
как ALLOW (допустимо) или DENY (заблокировать) перед вызовом основного агента.

Получает всю историю диалога, чтобы корректно оценивать контекст
(уточняющие вопросы в рамках on-topic диалога не должны блокироваться).
"""

from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from rag.config import (
    YANDEX_CLOUD_API_KEY,
    YANDEX_CLOUD_FOLDER,
    BASE_URL,
    GUARDRAIL_MODEL,
    GUARDRAIL_MAX_TOKENS,
    GUARDRAIL_TEMPERATURE,
    GUARDRAIL_SYSTEM_PROMPT,
)


@dataclass
class GuardrailResult:
    """Результат проверки guardrail."""
    allowed: bool
    reason: str  # "ALLOW" / "DENY" / текст ошибки


class Guardrail:
    """
    Классификатор входящих сообщений.

    Использует быструю модель (gpt-oss-20b) для определения,
    относится ли сообщение к теме кафедры и не является ли оно
    попыткой prompt injection / jailbreaking.
    """

    def __init__(self):
        print("🛡️  Инициализация Guardrail...")
        self.llm = ChatOpenAI(
            model=f"gpt://{YANDEX_CLOUD_FOLDER}/{GUARDRAIL_MODEL}",
            api_key="unused",
            openai_api_base=BASE_URL,
            temperature=GUARDRAIL_TEMPERATURE,
            max_tokens=GUARDRAIL_MAX_TOKENS,
            default_headers={
                "Authorization": f"Api-Key {YANDEX_CLOUD_API_KEY}",
            },
        )
        print("✅ Guardrail готов!")

    def check(
        self,
        question: str,
        history: list[dict] | None = None,
    ) -> GuardrailResult:
        """
        Проверяет сообщение пользователя.

        Args:
            question: Текущее сообщение пользователя.
            history: История диалога [{"role": "user"/"assistant", "content": "..."}].

        Returns:
            GuardrailResult с вердиктом (allowed=True/False).
        """
        # Формируем сообщения с историей
        messages = [SystemMessage(content=GUARDRAIL_SYSTEM_PROMPT)]

        if history:
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=question))

        try:
            response = self.llm.invoke(messages)
            verdict = (response.content or "").strip().upper()

            # Парсим ответ: если содержит "DENY" — блокируем,
            # всё остальное (включая пустой ответ) — пропускаем (fail-open)
            if "DENY" in verdict:
                return GuardrailResult(allowed=False, reason="DENY")
            else:
                return GuardrailResult(allowed=True, reason="ALLOW")

        except Exception as e:
            # При ошибке — пропускаем (fail-open)
            print(f"[Guardrail] Ошибка: {e}, пропускаем")
            return GuardrailResult(allowed=True, reason=f"ERROR: {e}")
