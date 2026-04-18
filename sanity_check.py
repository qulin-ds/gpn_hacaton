from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict
from collections import Counter

def analyze_md_file(md_path: Path) -> Dict:
    """Анализ одного MD файла с улучшенной проверкой таблиц."""
    try:
        content = md_path.read_text(encoding="utf-8")
    except Exception as e:
        return {"filename": md_path.name, "error": str(e)}

    lines = content.split('\n')
    
    # --- 1. Базовая статистика ---
    stats = {
        "filename": md_path.name,
        "size_kb": len(content.encode("utf-8")) / 1024,
        "char_count": len(content),
        "line_count": len(lines),
    }

    # --- 2. Структура (Заголовки) ---
    headers = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
    stats["headers_count"] = len(headers)
    stats["has_h1"] = any(h[0] == '#' for h in headers)
    
    # --- 3. Таблицы (Улучшенная логика) ---
    # Ищем блоки таблиц. Начало таблицы - строка с |, за которой следует строка с |---|
    # Мы будем искать все такие пары и смотреть, что идет дальше
    
    table_count = 0
    valid_table_count = 0
    
    # Разбиваем контент на потенциальные табличные блоки по пустым строкам или двойным переносам
    # Это упрощение, но для Markdown работает хорошо
    blocks = re.split(r'\n\s*\n', content)
    
    for block in blocks:
        if '|' not in block:
            continue
            
        block_lines = block.strip().split('\n')
        if len(block_lines) < 2:
            continue
            
        # Проверяем, есть ли разделитель |---|
        has_separator = False
        separator_index = -1
        
        for i, line in enumerate(block_lines):
            if re.match(r'^\|?[\s\-:|]+$', line) and '|' in line:
                has_separator = True
                separator_index = i
                break
                
        if has_separator and separator_index < len(block_lines) - 1:
            table_count += 1
            
            # Проверяем, есть ли данные ПОСЛЕ разделителя
            data_lines = block_lines[separator_index + 1:]
            has_data = False
            
            for d_line in data_lines:
                # Если в строке есть буквы или цифры вне служебных символов таблицы
                if re.search(r'[a-zA-Zа-яА-Я0-9]', d_line):
                    has_data = True
                    break
                    
            if has_data:
                valid_table_count += 1

    stats["tables_count"] = table_count
    stats["valid_tables"] = valid_table_count

    # --- 4. Изображения ---
    img_links = re.findall(r'!\[.*?\]\((.*?)\)', content)
    stats["images_linked"] = len(img_links)
    
    existing_imgs = 0
    missing_imgs = 0
    for link in img_links:
        clean_link = link.split('#')[0].strip()
        if not clean_link:
            continue
        img_path = md_path.parent / clean_link
        if img_path.exists():
            existing_imgs += 1
        else:
            missing_imgs += 1
            
    stats["images_existing"] = existing_imgs
    stats["images_missing"] = missing_imgs

    # --- 5. Качество текста (Garbage Detection) ---
    # Удаляем таблицы из текста для оценки чистоты prose
    text_no_tables = re.sub(r'\|.*?\|', '', content) 
    words = re.findall(r'[а-яА-Яa-zA-Z]{4,}', text_no_tables)
    
    if words:
        unique_ratio = len(set(words)) / len(words)
    else:
        unique_ratio = 0
        
    stats["text_unique_ratio"] = round(unique_ratio, 2)
    
    # Длинные строки без пробелов
    long_lines_no_space = sum(1 for line in lines if len(line) > 150 and ' ' not in line)
    stats["long_garbage_lines"] = long_lines_no_space

    stats["is_empty"] = stats["char_count"] < 100

    return stats


def main():
    parser = argparse.ArgumentParser(description="Universal Sanity Check v2")
    parser.add_argument("--input-dir", type=Path, required=True, help="Папка с результатами")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Ошибка: Папка {args.input_dir} не найдена.")
        return

    md_files = sorted(args.input_dir.glob("*.md"))
    if not md_files:
        print("Нет .md файлов.")
        return

    print(f"🔍 Анализ {len(md_files)} файлов...\n")
    
    all_stats = []
    anomalies = []

    for md_path in md_files:
        stats = analyze_md_file(md_path)
        all_stats.append(stats)
        
        reasons = []
        if stats.get("error"):
            reasons.append(f"❌ Ошибка: {stats['error']}")
        
        if stats["is_empty"]:
            reasons.append("⚠️ Файл почти пустой")
            
        if stats["images_linked"] > 0 and stats["images_missing"] > 0:
            pct = (stats["images_missing"] / stats["images_linked"]) * 100
            if pct > 50:
                reasons.append(f"🖼️ Потеряно {stats['images_missing']} из {stats['images_linked']} картинок")

        # Теперь жалуемся, только если таблиц МНОГО, но ни одна не валидна
        if stats["tables_count"] > 0 and stats["valid_tables"] == 0:
            reasons.append(f"📊 Обнаружено {stats['tables_count']} таблиц, но все кажутся пустыми/битыми")

        if stats["long_garbage_lines"] > 3:
            reasons.append(f"🧟 {stats['long_garbage_lines']} строк-монстров (слипшийся текст)")

        if stats["text_unique_ratio"] < 0.3 and stats["char_count"] > 500:
            reasons.append(f"🔄 Низное разнообразие слов ({stats['text_unique_ratio']}). Много повторов.")

        if reasons:
            anomalies.append({"file": stats["filename"], "reasons": reasons})

    # --- Вывод таблицы ---
    print(f"{'Файл':<25} | {'Кб':>5} | {'Заг.':>4} | {'Табл.':>5} | {'Валид.т.':>8} | {'Карт.':>5} | {'Уник.сл.':>8}")
    print("-" * 85)
    for s in all_stats[:15]:
        print(f"{s['filename']:<25} | {s['size_kb']:>5.1f} | {s['headers_count']:>4} | {s['tables_count']:>5} | {s['valid_tables']:>8} | {s['images_linked']:>5} | {s['text_unique_ratio']:>8}")
    
    if len(all_stats) > 15:
        print(f"... и еще {len(all_stats) - 15} файлов")

    # --- Итоги ---
    total_imgs = sum(s["images_linked"] for s in all_stats)
    total_miss = sum(s["images_missing"] for s in all_stats)
    total_valid_tables = sum(s["valid_tables"] for s in all_stats)
    
    print("\n" + "="*85)
    print("📊 ОБЩАЯ СТАТИСТИКА:")
    print(f"   Всего картинок: {total_imgs} (Потеряно: {total_miss})")
    print(f"   Всего таблиц: {sum(s['tables_count'] for s in all_stats)}")
    print(f"   Валидных таблиц (с данными): {total_valid_tables}")
    
    # --- Аномалии ---
    if anomalies:
        print("\n" + "!"*85)
        print(f"⚠️ НАЙДЕНО {len(anomalies)} ФАЙЛОВ ТРЕБУЮЩИХ ПРОВЕРКИ:")
        print("!"*85)
        for a in anomalies[:10]:
            print(f"\n📄 {a['file']}")
            for r in a['reasons']:
                print(f"   {r}")
        if len(anomalies) > 10:
            print(f"\n... и еще {len(anomalies) - 10} файлов с проблемами.")
    else:
        print("\n✅ Все файлы выглядят отлично! Проблем не обнаружено.")

if __name__ == "__main__":
    main()