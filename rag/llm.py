"""
LLM: обёртка над OpenRouter API для генерации ответов.

Использует openai-совместимый API OpenRouter.
Включает автоматический ретрай при rate-limit (429).
"""

import time

from openai import OpenAI, RateLimitError

from rag.config import (
    YANDEX_CLOUD_API_KEY,
    YANDEX_CLOUD_FOLDER,
    BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    SYSTEM_PROMPT,
)

# Максимум ретраев и начальная задержка (секунды)
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0


class LLM:
    """
    Обёртка над LLM через OpenRouter.

    Использует openai SDK (OpenRouter совместим с OpenAI API).
    При 429 (rate limit) автоматически ждёт и повторяет запрос.

    Использование:
        llm = LLM()
        answer = llm.ask("Кто заведует кафедрой?", context="...")
        print(answer)
    """

    def __init__(
        self,
        model=f"gpt://{YANDEX_CLOUD_FOLDER}/{LLM_MODEL}",
        api_key: str | None = None,
        base_url: str = BASE_URL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

        key = api_key or YANDEX_CLOUD_API_KEY
        if not key:
            raise ValueError(
                "API_KEY не задан! "
                "Укажите его в .env файле или передайте в конструктор."
            )

        self.client = OpenAI(
            api_key="unused",
            base_url=base_url,
            default_headers={
                "Authorization": f"Api-Key {key}",
            },
        )
        print(f"LLM инициализирован: {self.model}")

    def _build_messages(
        self,
        question: str,
        context: str,
        history: list[dict] | None = None,
    ) -> list[dict]:
        """
        Формирует список сообщений для LLM.

        Args:
            question: Вопрос пользователя.
            context: Контекст из ретривера.
            history: Опциональная история диалога — список
                     [{"role": "user"/"assistant", "content": "..."}].
        """
        user_message = (
            f"Контекст:\n{context}\n\n"
            f"Вопрос: {question}"
        )
        messages = [{"role": "system", "content": self.system_prompt}]

        # Добавляем историю диалога (если есть)
        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": user_message})
        return messages

    def ask(
        self,
        question: str,
        context: str,
        history: list[dict] | None = None,
    ) -> str:
        """
        Задать вопрос LLM с контекстом из ретривера.
        При 429 автоматически ретраит до MAX_RETRIES раз.

        Args:
            question: Вопрос пользователя.
            context: Контекст (полные тексты найденных страниц).
            history: Опциональная история диалога.

        Returns:
            Ответ модели (строка).
        """
        messages = self._build_messages(question, context, history)

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    extra_body={"reasoning_effort": "none"},
                )
                return response.choices[0].message.content.strip()

            except RateLimitError as e:
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"[LLM] Rate limit (429), ретрай через {delay:.0f}с... (попытка {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(delay)
                else:
                    raise RuntimeError(
                        f"LLM недоступна после {MAX_RETRIES} попыток (rate limit). "
                        f"Попробуйте позже."
                    ) from e

    def ask_stream(
        self,
        question: str,
        context: str,
        history: list[dict] | None = None,
    ):
        """
        Стриминговый вариант — отдаёт токены по мере генерации.
        При 429 автоматически ретраит.

        Args:
            question: Вопрос пользователя.
            context: Контекст из ретривера.
            history: Опциональная история диалога.

        Yields:
            Строковые токены по мере генерации.
        """
        messages = self._build_messages(question, context, history)

        for attempt in range(MAX_RETRIES + 1):
            try:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                    extra_body={"reasoning_effort": "none"},
                )

                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content
                return  # Успешно отстримили — выходим

            except RateLimitError as e:
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"[LLM] Rate limit (429), ретрай через {delay:.0f}с... (попытка {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(delay)
                else:
                    raise RuntimeError(
                        f"LLM недоступна после {MAX_RETRIES} попыток (rate limit). "
                        f"Попробуйте позже."
                    ) from e
