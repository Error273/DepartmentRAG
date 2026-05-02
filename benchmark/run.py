"""
Бенчмарк RAG-агента с LLM-as-Judge.

Пайплайн:
  1. Читает questions.csv (question;source)
  2. Прогоняет каждый вопрос через RAG-агента
  3. Загружает полный текст эталонного документа из doc_texts.json
  4. Отдаёт LLM-судье (вопрос, ответ агента, текст документа)
  5. Судья выставляет score (y/n) и пишет комментарий
  6. Результат сохраняется в benchmark/results.csv

Использование:
    python -m benchmark.run
    python -m benchmark.run --output benchmark/results.csv
"""

import csv
import json
import sys
import time
import argparse
from pathlib import Path

# UTF-8 для Windows
sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from rag.agent import RAGAgent
from rag.config import (
    YANDEX_CLOUD_API_KEY,
    YANDEX_CLOUD_FOLDER,
    BASE_URL,
    LLM_MODEL,
    DOC_TEXTS_PATH,
)

# ── Промпт для LLM-судьи ────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """\
Ты — строгий и объективный судья качества ответов RAG-системы.

Тебе дают:
- Вопрос пользователя
- Ответ RAG-агента
- Полный текст эталонного документа, в котором содержится правильный ответ

Твоя задача — оценить, правильно ли ответил агент.

Правила оценки:
1. Ответ считается ПРАВИЛЬНЫМ (y), если он содержит корректную фактическую информацию, \
которая подтверждается текстом документа. Допускаются незначительные отклонения \
в формулировках — важна суть.
2. Ответ считается НЕПРАВИЛЬНЫМ (n), если:
   - Содержит фактические ошибки
   - Не отвечает на заданный вопрос
   - Содержит выдуманную информацию (галлюцинации), которой нет в документе
   - Агент не смог найти информацию, хотя она есть в документе
3. Если ответ частично правильный, но содержит существенные пропуски ключевых фактов \
или фактические ошибки — ставь "n".
4. Если ответ получен с помощью других документов и содержит верный ответ на вопрос, ставь "y"

Формат ответа — СТРОГО две строки:
score: y
comment: <краткий комментарий на русском, 1-2 предложения>

или

score: n
comment: <краткий комментарий на русском, 1-2 предложения, что именно не так>
"""


def build_judge_prompt(question: str, agent_answer: str, doc_text: str) -> str:
    """Формирует промпт для LLM-судьи."""
    # Ограничиваем длину документа, чтобы не вылезти из контекста
    max_doc_len = 12000
    if len(doc_text) > max_doc_len:
        doc_text = doc_text[:max_doc_len] + "\n... [текст обрезан]"

    return (
        f"## Вопрос\n{question}\n\n"
        f"## Ответ RAG-агента\n{agent_answer}\n\n"
        f"## Текст эталонного документа\n{doc_text}"
    )


def parse_judge_response(response_text: str) -> tuple[str, str]:
    """
    Парсит ответ судьи. Возвращает (score, comment).
    Если не удалось распарсить — возвращает ("?", raw_text).
    """
    score = "?"
    comment = response_text.strip()

    for line in response_text.strip().split("\n"):
        line_lower = line.strip().lower()
        if line_lower.startswith("score:"):
            val = line_lower.split(":", 1)[1].strip()
            if val in ("y", "n"):
                score = val
        elif line_lower.startswith("comment:"):
            comment = line.split(":", 1)[1].strip()

    return score, comment


def load_doc_texts(path: Path) -> dict:
    """Загружает полные тексты документов."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_questions(path: Path) -> list[dict]:
    """Читает questions.csv (разделитель ;)."""
    questions = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            q = row.get("question", "").strip()
            src = row.get("source", "").strip()
            if q and src:
                questions.append({"question": q, "source": src})
    return questions


def find_doc_text(source: str, doc_texts: dict) -> str:
    """
    Ищет текст документа по source.
    Source может быть URL (https://...) или именем файла (pudovkin_100.doc).
    В doc_texts.json ключи — URL или local://docs/<filename>.
    """
    # Попробуем прямое совпадение
    if source in doc_texts:
        return doc_texts[source].get("text", "")

    # Для локальных файлов: ищем по имени файла в ключах local://docs/
    source_lower = source.lower()
    for key, val in doc_texts.items():
        if key.startswith("local://docs/"):
            filename = key.replace("local://docs/", "")
            if filename.lower() == source_lower:
                return val.get("text", "")

    # Не нашли точное совпадение — пробуем нечёткое (имя без расширения)
    source_stem = Path(source).stem.lower()
    for key, val in doc_texts.items():
        key_stem = Path(key.replace("local://docs/", "")).stem.lower()
        if key_stem == source_stem:
            return val.get("text", "")

    return ""


def main():
    parser = argparse.ArgumentParser(description="Бенчмарк RAG-агента с LLM-as-Judge")
    parser.add_argument(
        "--questions",
        type=str,
        default=str(PROJECT_ROOT / "benchmark" / "questions.csv"),
        help="Путь к CSV с вопросами",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "benchmark" / "results.csv"),
        help="Путь для сохранения результатов",
    )
    args = parser.parse_args()

    # 1. Загружаем данные
    questions = load_questions(Path(args.questions))[:3]
    if not questions:
        print("❌ Нет вопросов в файле")
        sys.exit(1)

    doc_texts = load_doc_texts(DOC_TEXTS_PATH)
    print(f"📋 Загружено {len(questions)} вопросов")
    print(f"📚 Загружено {len(doc_texts)} документов")

    # 2. Инициализируем RAG-агента
    print("\n🤖 Инициализация RAG-агента...")
    agent = RAGAgent()

    # 3. Инициализируем LLM-судью (та же модель)
    print("⚖️  Инициализация LLM-судьи...")
    judge_llm = ChatOpenAI(
        model=f"gpt://{YANDEX_CLOUD_FOLDER}/{LLM_MODEL}",
        api_key="unused",
        openai_api_base=BASE_URL,
        temperature=0,
        max_tokens=512,
        default_headers={
            "Authorization": f"Api-Key {YANDEX_CLOUD_API_KEY}",
        },
        model_kwargs={"reasoning_effort": "none"},
    )

    # 4. Прогоняем бенчмарк
    run_start = time.time()
    results = []
    correct = 0
    total = len(questions)

    print(f"\n{'=' * 70}")
    print(f"  BENCHMARK: LLM-as-Judge ({LLM_MODEL})")
    print(f"  Всего вопросов: {total}")
    print(f"{'=' * 70}\n")

    for i, q in enumerate(questions, 1):
        question = q["question"]
        source = q["source"]

        # 4a. Получаем ответ агента
        print(f"  [{i:2d}/{total}] Вопрос: {question}")
        print(f"           Источник: {source}")

        try:
            start_time = time.time()
            response = agent.ask(question)
            agent_elapsed = time.time() - start_time
            agent_answer = response.answer
            print(f"           Агент ответил за {agent_elapsed:.1f}s")
        except Exception as e:
            agent_answer = f"[ОШИБКА АГЕНТА: {e}]"
            print(f"           ❌ Ошибка агента: {e}")

        # 4b. Находим текст эталонного документа
        doc_text = find_doc_text(source, doc_texts)
        if not doc_text:
            print(f"           ⚠️  Текст документа не найден для: {source}")
            results.append({
                "question": question,
                "source": source,
                "answer": agent_answer,
                "score": "?",
                "comment": "Текст эталонного документа не найден",
            })
            continue

        # 4c. Оценка LLM-судьёй (с ретраем при пустом/непарсируемом ответе)
        judge_prompt = build_judge_prompt(question, agent_answer, doc_text)
        score, comment = "?", ""
        max_judge_retries = 2

        for attempt in range(max_judge_retries + 1):
            try:
                judge_response = judge_llm.invoke([
                    SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                    HumanMessage(content=judge_prompt),
                ])
                raw = judge_response.content or ""
                score, comment = parse_judge_response(raw)

                if score != "?":
                    break  # Успешно распарсили

                # Не удалось распарсить — логируем и ретраим
                print(f"           ⚠️  Судья вернул непарсируемый ответ (попытка {attempt + 1}): {raw[:200]}")
                if attempt < max_judge_retries:
                    time.sleep(1)

            except Exception as e:
                score, comment = "?", f"Ошибка судьи: {e}"
                print(f"           ❌ Ошибка судьи (попытка {attempt + 1}): {e}")
                if attempt < max_judge_retries:
                    time.sleep(1)

        if score == "y":
            correct += 1
            icon = "✅"
        elif score == "n":
            icon = "❌"
        else:
            icon = "❓"

        print(f"           {icon} Оценка: {score} | {comment}")
        print()

        results.append({
            "question": question,
            "source": source,
            "answer": agent_answer,
            "score": score,
            "comment": comment,
        })

    # 5. Сохраняем результаты
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "source", "answer", "score", "comment"],
            delimiter=";",
        )
        writer.writeheader()
        writer.writerows(results)

    # 6. Итоги
    answered = sum(1 for r in results if r["score"] in ("y", "n"))
    errors = sum(1 for r in results if r["score"] == "?")

    print(f"{'=' * 70}")
    print(f"  РЕЗУЛЬТАТЫ")
    print(f"{'=' * 70}")
    print(f"  Accuracy:  {correct}/{answered} ({correct/answered:.1%})" if answered else "  Accuracy:  N/A")
    print(f"  Correct:   {correct}")
    print(f"  Incorrect: {answered - correct}")
    if errors:
        print(f"  Ошибки:    {errors}")
    print(f"\n  Результаты сохранены: {output_path}")
    print(f"\n  Время выполнения: {time.time() - run_start}s")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
