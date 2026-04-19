from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict
from collections import Counter

def analyze_md_file(md_path: Path) -> Dict:
    """Анализ одного MD файла."""
    try:
        content = md_path.read_text(encoding="utf-8")
    except Exception as e:
        return {"filename": md_path.name, "error": str(e)}

    lines = content.split('\n')
    
    stats = {
        "filename": md_path.name,
        "size_kb": len(content.encode("utf-8")) / 1024,
        "char_count": len(content),
    }

    # Заголовки
    headers = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
    stats["headers_count"] = len(headers)
    
    # Таблицы (упрощенно)
    table_count = len(re.findall(r'\|[\s\-:|]+\|', content))
    stats["tables_count"] = table_count

    # Изображения
    img_links = re.findall(r'!\[.*?\]\((.*?)\)', content)
    stats["images_linked"] = len(img_links)
    
    existing_imgs = 0
    missing_imgs = 0
    for link in img_links:
        clean_link = link.split('#')[0].strip()
        if not clean_link: continue
        img_path = md_path.parent / clean_link
        if img_path.exists():
            existing_imgs += 1
        else:
            missing_imgs += 1
            
    stats["images_existing"] = existing_imgs
    stats["images_missing"] = missing_imgs

    # Мусор (длинные строки без пробелов)
    long_lines_no_space = sum(1 for line in lines if len(line) > 150 and ' ' not in line)
    stats["long_garbage_lines"] = long_lines_no_space

    return stats


def main():
    parser = argparse.ArgumentParser(description="Sanity Check")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--max-files", "-n", type=int, default=None, help="Количество файлов для проверки")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Ошибка: Папка {args.input_dir} не найдена.")
        return

    md_files = sorted(args.input_dir.glob("*.md"))
    if not md_files:
        print("Нет .md файлов.")
        return

    if args.max_files is not None:
        md_files = md_files[:args.max_files]
        print(f"🔍 Анализ первых {len(md_files)} файлов...\n")
    else:
        print(f"🔍 Анализ всех {len(md_files)} файлов...\n")
    
    anomalies = []
    total_miss = 0
    total_imgs = 0

    for md_path in md_files:
        stats = analyze_md_file(md_path)
        total_imgs += stats["images_linked"]
        total_miss += stats["images_missing"]
        
        reasons = []
        if stats["images_missing"] > 0:
            reasons.append(f"🖼️ Потеряно {stats['images_missing']} картинок")
        if stats["long_garbage_lines"] > 3:
            reasons.append(f"🧟 {stats['long_garbage_lines']} строк-монстров")
            
        if reasons:
            anomalies.append({"file": stats["filename"], "reasons": reasons})
            
        # Вывод статистики по каждому файлу (первые 10)
        if md_files.index(md_path) < 10:
             print(f"{stats['filename']:<25} | Карт: {stats['images_linked']} (Miss: {stats['images_missing']}) | Табл: {stats['tables_count']}")

    print("\n" + "="*70)
    print(f"Всего картинок: {total_imgs}, Потеряно: {total_miss}")
    
    if anomalies:
        print(f"\n⚠️ Найдено {len(anomalies)} файлов с проблемами:")
        for a in anomalies[:5]:
            print(f"  - {a['file']}: {', '.join(a['reasons'])}")
    else:
        print("\n✅ Проблем не обнаружено.")

if __name__ == "__main__":
    main()