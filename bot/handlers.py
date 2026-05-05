"""
Обработчики команд и сообщений Telegram-бота (aiogram 3).

Бот работает напрямую с RAGPipeline — без FastAPI-посредника.
Поддерживает память диалога: LLM видит предыдущие вопросы и ответы.
"""

import re
import asyncio
import traceback
from collections import defaultdict, deque

from aiogram import Router, F
from aiogram.types import Message
from aiogram.filters import CommandStart, Command
from aiogram.enums import ParseMode

from rag.pipeline import get_pipeline


router = Router()

# ── Конфигурация памяти ──────────────────────────────────────────────

# Сколько пар (вопрос-ответ) хранить на каждый чат.
# 10 пар = 20 сообщений в истории → достаточно для контекста,
# но не перегружает контекстное окно LLM.
MAX_HISTORY_PAIRS = 10

# Хранилище: chat_id → deque[{"role": ..., "content": ...}]
_chat_history: dict[int, deque] = defaultdict(
    lambda: deque(maxlen=MAX_HISTORY_PAIRS * 2)
)





# ── Утилиты ──────────────────────────────────────────────────────────

def md_to_html(text: str) -> str:
    """
    Конвертирует Markdown-ответ LLM в Telegram-совместимый HTML.

    Поддерживает: **bold**, *italic*, `code`, ```блоки```,
    [ссылки](url), заголовки (#, ##, ###).
    """
    # 1. Экранируем HTML-сущности
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")

    # 2. Блоки кода ```...``` → <pre><code>...</code></pre>
    text = re.sub(
        r"```(?:\w*)?\n?(.*?)```",
        r"<pre><code>\1</code></pre>",
        text,
        flags=re.DOTALL,
    )

    # 3. Инлайн-код `...` → <code>...</code>
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)

    # 4. Жирный **text** или __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)

    # 5. Курсив *text* или _text_ (но не внутри слов с _)
    text = re.sub(r"(?<!\w)\*([^*]+?)\*(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!\w)_([^_]+?)_(?!\w)", r"<i>\1</i>", text)

    # 6. Ссылки [text](url)
    text = re.sub(r"\[([^\]]+)]\(([^)]+)\)", r'<a href="\2">\1</a>', text)

    # 7. Заголовки ### → жирный текст
    text = re.sub(r"^#{1,3}\s+(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)

    return text


def format_sources(sources) -> str:
    """Форматирует список источников в читаемый текст."""
    if not sources:
        return ""

    lines = ["\n\n📚 <b>Источники:</b>"]
    seen = set()
    for doc in sources:
        if doc.source_url in seen:
            continue
        seen.add(doc.source_url)
        title = doc.title or "Без названия"
        lines.append(
            f'  • <a href="{doc.source_url}">{title}</a>'
            f" (релевантность: {doc.score:.2f})"
        )
    return "\n".join(lines)


def format_tool_logs(tool_logs, elapsed_seconds: float = 0.0, total_tokens: int = 0) -> str:
    """Форматирует логи вызовов инструментов в читаемый текст."""
    if not tool_logs:
        return ""

    lines = ["🔧 <b>Логи работы агента:</b>\n"]

    # Метрики производительности
    metrics = []
    if elapsed_seconds > 0:
        metrics.append(f"⏱ Время: <code>{elapsed_seconds:.1f}с</code>")
    if total_tokens > 0:
        metrics.append(f"🔤 Токены: <code>{total_tokens}</code>")
    if metrics:
        lines.append(" | ".join(metrics))
        lines.append("")

    for i, log in enumerate(tool_logs, 1):
        lines.append(f"<b>Шаг {i}:</b> 🛠 <code>{log.tool_name}</code>")

        # Аргументы
        args_parts = []
        for key, value in log.arguments.items():
            args_parts.append(f"  • <i>{key}</i>: <code>{value}</code>")
        if args_parts:
            lines.append("\n".join(args_parts))

        # Результат (превью)
        if log.result:
            # Обрезаем для читаемости в Telegram
            result_preview = log.result[:300]
            if len(log.result) > 300:
                result_preview += "..."
            lines.append(f"  ➡️ Результат: <pre>{result_preview}</pre>")
        else:
            lines.append(f"  ➡️ Результат: <i>нет данных</i>")

        lines.append("")  # пустая строка между шагами

    return "\n".join(lines)


# ── /start ───────────────────────────────────────────────────────────

START_TEXT = (
    "👋 <b>Привет!</b>\n\n"
    "Я — интеллектуальный помощник <b>кафедры аэрогидромеханики КФУ</b>.\n\n"
    "Задайте мне любой вопрос о кафедре, и я найду ответ, "
    "используя информацию с официального сайта.\n\n"
    "📝 <b>Примеры вопросов:</b>\n"
    "  • Кто заведует кафедрой?\n"
    "  • Какие дисциплины преподают?\n"
    "  • Какие научные направления у кафедры?\n\n"
    "💡 Просто напишите вопрос в чат!\n\n"
    "🧠 Я запоминаю диалог — можете задавать уточняющие вопросы.\n"
    "Команда /clear очистит историю."
)


@router.message(CommandStart())
async def cmd_start(message: Message):
    """Обработка команды /start."""
    # Сбрасываем историю при /start
    _chat_history[message.chat.id].clear()
    await message.answer(START_TEXT, parse_mode=ParseMode.HTML)


# ── /help ────────────────────────────────────────────────────────────

HELP_TEXT = (
    "ℹ️ <b>Как пользоваться ботом</b>\n\n"
    "1. Напишите вопрос о кафедре аэрогидромеханики\n"
    "2. Бот найдёт релевантную информацию на сайте кафедры\n"
    "3. Сгенерирует подробный ответ с указанием источников\n\n"
    "<b>Команды:</b>\n"
    "/start — начать заново\n"
    "/help — эта справка\n"
    "/clear — очистить память диалога\n\n"
    "🧠 <b>Память:</b> бот помнит контекст разговора, "
    "поэтому вы можете задавать уточняющие вопросы.\n\n"
    "⚠️ Бот отвечает только на вопросы о кафедре. "
    "Если информации нет на сайте — бот честно скажет об этом."
)


@router.message(Command("help"))
async def cmd_help(message: Message):
    """Обработка команды /help."""
    await message.answer(HELP_TEXT, parse_mode=ParseMode.HTML)


# ── /clear ───────────────────────────────────────────────────────────

@router.message(Command("clear"))
async def cmd_clear(message: Message):
    """Очистка истории диалога."""
    _chat_history[message.chat.id].clear()
    await message.answer(
        "🗑 История диалога очищена. Можете начать новый разговор!",
    )


# ── Отправка логов инструментов ───────────────────────────────────────

async def send_tool_logs(message: Message, response) -> None:
    """
    Отправляет логи вызовов инструментов отдельным сообщением.

    Вынесена в отдельную функцию для переиспользования.
    Вызовите из хэндлера, передав исходное сообщение пользователя
    и объект ответа RAGPipeline.
    """
    tool_logs = getattr(response, 'tool_logs', None)
    if not tool_logs:
        return

    logs_text = format_tool_logs(
        tool_logs,
        elapsed_seconds=getattr(response, 'elapsed_seconds', 0.0),
        total_tokens=getattr(response, 'total_tokens', 0),
    )
    if not logs_text:
        return

    try:
        await message.answer(
            logs_text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
    except Exception:
        # Фоллбэк без HTML
        plain_logs = "🔧 Логи работы агента:\n\n"
        for j, lg in enumerate(tool_logs, 1):
            plain_logs += f"Шаг {j}: {lg.tool_name}\n"
            for k, v in lg.arguments.items():
                plain_logs += f"  {k}: {v}\n"
            if lg.result:
                plain_logs += f"  Результат: {lg.result[:200]}\n"
            plain_logs += "\n"
        await message.answer(
            plain_logs,
            disable_web_page_preview=True,
        )


# ── Обработка вопросов (основная логика) ─────────────────────────────

@router.message(F.text)
async def handle_question(message: Message):
    """
    Обработка текстовых сообщений — основной RAG-цикл с памятью.

    1. Показывает статус «печатает...»
    2. Находит релевантные документы
    3. Генерирует ответ LLM с учётом истории диалога
    4. Сохраняет вопрос и ответ в память
    5. Отправляет ответ с источниками
    """
    question = message.text.strip()
    chat_id = message.chat.id

    if not question:
        await message.answer("❓ Пожалуйста, задайте вопрос.")
        return

    try:
        pipeline = get_pipeline()

        # Получаем историю диалога для этого чата
        history = list(_chat_history[chat_id])

        # Показываем статус «печатает» пока агент работает
        await message.bot.send_chat_action(chat_id, "typing")

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: pipeline.ask(
                question=question,
                history=history,
            ),
        )

        answer = response.answer
        docs = response.sources

        # Если запрос заблокирован guardrail-ом — отправляем отказ
        if response.blocked:
            await message.answer(
                f"🛡️ {answer}",
                parse_mode=ParseMode.HTML,
            )
            return

        # Сохраняем в память диалога (только разрешённые сообщения)
        _chat_history[chat_id].append({"role": "user", "content": question})
        _chat_history[chat_id].append({"role": "assistant", "content": answer})

        # Формируем финальное сообщение (MD → HTML)
        final_text = md_to_html(answer)

        try:
            await message.answer(
                final_text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
        except Exception:
            # Фоллбэк без HTML, если парсинг сломался
            await message.answer(
                answer,
                disable_web_page_preview=True,
            )

    except Exception as e:
        traceback.print_exc()
        await message.answer(
            f"⚠️ Произошла ошибка при обработке вашего вопроса.\n\n"
            f"Детали: {e}\n\n"
            f"Попробуйте ещё раз через несколько секунд."
        )
