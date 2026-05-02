"""
Скрипт чанковки: читает data/cleaned/, разбивает на чанки, сохраняет в data/chunks/chunks.json
"""

import json
import os
import sys

from rag.chunker import chunk_document

sys.stdout.reconfigure(encoding='utf-8')

CLEANED_DIR = '../data/cleaned'
CHUNKS_DIR = '../data/chunks'
CHUNKS_FILE = os.path.join(CHUNKS_DIR, 'chunks.json')

# Параметры чанковки
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def detect_category(rel_path: str) -> str:
    """Определяет категорию по пути к файлу."""
    if rel_path.startswith('news'):
        return 'news'
    elif rel_path.startswith('people'):
        return 'people'
    elif rel_path.startswith('docs'):
        return 'docs'
    else:
        return 'main'


def main():
    all_chunks = []
    file_count = 0

    for root, dirs, files in os.walk(CLEANED_DIR):
        for filename in sorted(files):
            if not filename.endswith('.json'):
                continue

            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, CLEANED_DIR)
            category = detect_category(rel_path)

            with open(filepath, 'r', encoding='utf-8') as f:
                doc = json.load(f)

            chunks = chunk_document(doc, CHUNK_SIZE, CHUNK_OVERLAP, category)
            all_chunks.extend(chunks)
            file_count += 1

            print(f'[{category:6}] {rel_path:<75} -> {len(chunks)} чанков')

    # Сохраняем
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    with open(CHUNKS_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f'\n{"=" * 80}')
    print(f'Файлов обработано: {file_count}')
    print(f'Всего чанков: {len(all_chunks)}')
    print(f'Средний размер чанка: {sum(len(c["text"]) for c in all_chunks) / len(all_chunks):.0f} символов')
    print(f'Параметры: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}')
    print(f'Сохранено в: {os.path.abspath(CHUNKS_FILE)}')


if __name__ == '__main__':
    main()
