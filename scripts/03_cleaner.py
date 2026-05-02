"""
Скрипт очистки спарсенных данных сайта КФУ.
Удаляет мусор из поля content: навигацию, хлебные крошки,
куки-баннер, форму логина, боковое меню и т.д.
Результат сохраняется в cleaned/ с сохранением структуры папок.
"""

import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

PARSED_DATA_DIR = '../data/raw/'
CLEARED_DATA_DIR = '../data/cleaned/'


# ============================================================
# Паттерны мусора, которые нужно удалить
# ============================================================

# 1. Хлебные крошки — удаляем через строковой поиск (см. функцию remove_breadcrumbs)
#    Слишком хрупко через regex из-за экранирования обратных слэшей

# 2. Cookie-баннер + форма входа (всё от "Для улучшения работы сайта" до конца)
COOKIE_FOOTER_PATTERN = re.compile(
    r'Для улучшения работы сайта и его взаимодействия с пользователями мы используем файлы cookie\.'
    r'.*$',
    re.DOTALL
)

# 3. Навигационное меню кафедры (боковая панель)
SIDEBAR_NAV_PATTERN = re.compile(
    r'История кафедры\s+Состав кафедры\s+Методические пособия\s+'
    r'Семинары и кружки\s+Темы курсовых и дипломных работ\s+'
    r'Электронные\s+ресурсы\s+Публикации\s+Встречи с работодателями\s+'
    r'График консультаций\s+АБИТУРИЕНТУ о бакалавриате\s+АБИТУРИЕНТУ о магистратуре'
    r'(\s+Фотоотчеты)?'  # иногда есть
)

# 4. Блок "Другие новости и объявления" с датами и заголовками
OTHER_NEWS_PATTERN = re.compile(
    r'(Подпишитесь на нашу рассылку\s+)?'
    r'Другие новости и объявления\s+'
    r'(\d{2}\s+\S+\s+\d{4}\s+.+?)(?=Для улучшения|$)',
    re.DOTALL
)

# 5. Блок "Все новости" (на главной)
ALL_NEWS_LINK = re.compile(r'\s*Все новости\s*')

# 6. Блок "Все объявления" (на главной)
ALL_ANNOUNCEMENTS_LINK = re.compile(r'\s*Все объявления\s*')

# 7. Категории-теги новостей
CATEGORY_TAGS_PATTERN = re.compile(
    r'\d+\s+'  # число просмотров
    r'Образование\s+Наука\s+Конкурсы\s+Студенческая жизнь\s+'
    r'Международное\s+Культура\s+Спорт\s+Сотрудничество'
)

# Альтернативный паттерн (сокращенный набор тегов)
CATEGORY_TAGS_SHORT = re.compile(
    r'\d+\s+(?:Образование\s+)?(?:Наука\s+)?(?:Конкурсы\s+)?'
    r'(?:Студенческая жизнь\s+)?(?:Международное\s+)?'
    r'(?:Культура\s+)?(?:Спорт\s+)?(?:Сотрудничество\s+)?'
    r'Подпишитесь на нашу рассылку'
)

# 8. Меню персональной страницы сотрудника
PERSONAL_PAGE_NAV = re.compile(
    r'Общие сведения\s+Направления научной работы\s+Преподаваемые дисциплины\s+'
    r'Показатели продуктивности\s+Повышение квалификации\s+'
    r'Награды,\s*почетные звания\s+Расписание\s+Мероприятия\s+'
    r'Результативность работы\s+Монографии\s+Статьи\s+'
    r'Тезисы и материалы конференций\s+Учебники и учебные пособия\s+'
    r'Патенты,\s*лицензии и тд\.\s+Электронные образовательные ресурсы\s+'
    r'Руководство НИР\s+Подготовка кадров\s+Публикации в СМИ\s+'
    r'Дистанционное образование'
)

# 9. Фото-подпись на главной
PHOTO_CAPTION = re.compile(r'Фотография из материала:\s*\[\.{3}\]')


def remove_breadcrumbs(content: str) -> str:
    """Удаляет хлебные крошки строковым поиском (надёжнее regex для обратных слэшей)."""
    # Ищем начало хлебных крошек
    marker_start = 'Главная'
    # Ищем конец — "Отделение механики" или "Кафедра аэрогидромеханики" + возможный подраздел
    marker_end_options = [
        'Кафедра аэрогидромеханики',
        'Отделение механики',
    ]

    idx_start = content.find(marker_start)
    if idx_start == -1:
        return content

    # Ищем конец хлебных крошек: последний из маркеров, который идёт после "Главная"
    best_end = -1
    for marker in marker_end_options:
        idx = content.find(marker, idx_start)
        if idx != -1:
            end_pos = idx + len(marker)
            if end_pos > best_end:
                best_end = end_pos

    if best_end == -1:
        return content

    # Проверяем, есть ли ещё один подраздел после "Кафедра аэрогидромеханики"
    # Например: "\ График консультаций" или "\ Семинары и кружки"
    remaining = content[best_end:]
    # Смотрим, начинается ли продолжение с " \ Название"
    sub_match = re.match(r'\s*\\\s*(\S[^\\]{2,}?)(?=\s{2,}|\s*$)', remaining)
    if sub_match:
        # Это подраздел навигации, удаляем и его
        best_end += sub_match.end()

    content = content[:idx_start] + content[best_end:]
    return content


def remove_keywords_before_nav(content: str) -> str:
    """Удаляет 'Ключевые слова: ...' если они склеены с навигацией кафедры."""
    # Паттерн: "Ключевые слова: что-то что-то История кафедры"
    # Нужно убрать "Ключевые слова: ..." до "История кафедры"
    pattern = re.compile(r'Ключевые слова:\s*.+?(?=\s+История кафедры)', re.DOTALL)
    return pattern.sub('', content)


def clean_content(content: str, title: str) -> str:
    """Очищает content от мусорных элементов сайта."""

    # Убираем дубликат заголовка в начале content
    # Заголовок часто повторяется дословно в начале (включая суффиксы через \)
    if title:
        if content.startswith(title):
            content = content[len(title):].strip()
        else:
            # Попробуем убрать title без суффиксов ("Семинары и кружки" из "Семинары и кружки\Кафедра...")
            clean_title_part = title.split('\\')[0].strip()
            if clean_title_part and content.startswith(clean_title_part + '\\'):
                # Убираем полный заголовок с суффиксом из content
                # Ищем конец заголовочной части (до слова "Главная")
                idx = content.find('Главная')
                if idx > 0:
                    content = content[idx:].strip()
                else:
                    content = content[len(clean_title_part):].strip()

    # 1. Хлебные крошки (строковой поиск — надёжнее regex)
    content = remove_breadcrumbs(content)

    # 2. Блок "Другие новости и объявления"
    content = OTHER_NEWS_PATTERN.sub('', content)

    # 3. Категории-теги
    content = CATEGORY_TAGS_PATTERN.sub('', content)
    content = CATEGORY_TAGS_SHORT.sub('', content)

    # 4. Ключевые слова перед навигацией
    content = remove_keywords_before_nav(content)

    # 5. Навигационное меню кафедры
    content = SIDEBAR_NAV_PATTERN.sub('', content)

    # 6. Меню персональной страницы
    content = PERSONAL_PAGE_NAV.sub('', content)

    # 7. "Все новости" / "Все объявления"
    content = ALL_NEWS_LINK.sub(' ', content)
    content = ALL_ANNOUNCEMENTS_LINK.sub(' ', content)

    # 8. Фото-подпись
    content = PHOTO_CAPTION.sub('', content)

    # 9. Cookie-баннер + форма входа (всегда в конце — удаляем последним)
    content = COOKIE_FOOTER_PATTERN.sub('', content)

    # Финальная чистка
    content = re.sub(r'\s{3,}', '  ', content)  # схлопываем большие пробелы
    content = content.strip()

    return content


def clean_title(title: str) -> str:
    """Убирает суффиксы навигации из title."""
    # "Семинары и кружки\Кафедра аэрогидромеханики - Казанский (Приволжский) федеральный университет"
    # -> "Семинары и кружки"
    suffixes_to_remove = [
        '\\Кафедра аэрогидромеханики - Казанский (Приволжский) федеральный университет',
        '\\Отделение механики - Казанский (Приволжский) федеральный университет',
        '. Персональная страница сотрудника КФУ. Казанский (Приволжский) федеральный университет.',
        '. Общие сведения. Персональная страница сотрудника КФУ. Казанский (Приволжский) федеральный университет.',
    ]
    for suffix in suffixes_to_remove:
        if suffix in title:
            title = title.split(suffix)[0].strip()

    return title


def process_file(src_path: str, dst_path: str) -> dict:
    """Обрабатывает один JSON файл: чистит и сохраняет."""
    with open(src_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    original_content = data.get('content', '')
    original_title = data.get('title', '')

    cleaned = {
        'url': data.get('url', ''),
        'title': clean_title(original_title),
        'content': clean_content(original_content, original_title),
    }

    # Сохраняем другие поля, если есть
    for key in data:
        if key not in cleaned:
            cleaned[key] = data[key]

    # Создаём директорию, если нужно
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    with open(dst_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    # Возвращаем статистику
    return {
        'file': os.path.relpath(src_path, PARSED_DATA_DIR),
        'original_len': len(original_content),
        'cleaned_len': len(cleaned['content']),
        'removed': len(original_content) - len(cleaned['content']),
    }


def main():
    stats = []

    for root, dirs, files in os.walk(PARSED_DATA_DIR):
        for filename in sorted(files):
            if not filename.endswith('.json'):
                continue

            src_path = os.path.join(root, filename)
            rel_path = os.path.relpath(src_path, PARSED_DATA_DIR)
            dst_path = os.path.join(CLEARED_DATA_DIR, rel_path)

            try:
                stat = process_file(src_path, dst_path)
                stats.append(stat)
                pct = (stat['removed'] / stat['original_len'] * 100) if stat['original_len'] > 0 else 0
                print(f"[OK] {stat['file']:<75} {stat['original_len']:>6} -> {stat['cleaned_len']:>6}  ({pct:5.1f}% удалено)")
            except Exception as e:
                print(f"[ERR] {rel_path}: {e}")

    print('\n' + '=' * 110)
    total_orig = sum(s['original_len'] for s in stats)
    total_clean = sum(s['cleaned_len'] for s in stats)
    total_removed = total_orig - total_clean
    pct = (total_removed / total_orig * 100) if total_orig > 0 else 0

    print(f"Файлов обработано: {len(stats)}")
    print(f"Общий размер content до очистки:    {total_orig:>8} символов")
    print(f"Общий размер content после очистки:  {total_clean:>8} символов")
    print(f"Удалено:                             {total_removed:>8} символов ({pct:.1f}%)")
    print(f"\nРезультат сохранен в: {os.path.abspath(CLEARED_DATA_DIR)}")


if __name__ == '__main__':
    main()
