#!/usr/bin/env python3
"""
Метрики для оценки качества парсинга PDF в Markdown.
Взвешенная сумма 4 компонент:
1. Таблицы (приоритет 1)
2. Текст (приоритет 2)
3. Структура (приоритет 3)
4. Изображения (приоритет 4)
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import argparse
import Levenshtein

# ============================================================================
# Веса компонент метрики
# ============================================================================
WEIGHTS = {
    'tables': 0.35,      # Таблицы - высший приоритет
    'text': 0.30,         # Текст - второй приоритет
    'structure': 0.20,    # Структура - третий приоритет
    'images': 0.15,       # Изображения - четвертый приоритет
}

# ============================================================================
# Вспомогательные функции
# ============================================================================

def normalize_text(text: str) -> str:
    """Нормализует текст для сравнения."""
    # Приводим к нижнему регистру
    text = text.lower()
    # Заменяем множественные пробелы и переносы строк
    text = re.sub(r'\s+', ' ', text)
    # Удаляем пробелы вокруг знаков препинания
    text = re.sub(r'\s*([.,;:!?])\s*', r'\1 ', text)
    return text.strip()

def levenshtein_similarity(s1: str, s2: str) -> float:
    """Нормализованное сходство Левенштейна (0..1)."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    s1 = normalize_text(s1)
    s2 = normalize_text(s2)
    
    distance = Levenshtein.distance(s1, s2)
    max_len = max(len(s1), len(s2))
    
    return 1.0 - (distance / max_len)

# ============================================================================
# Извлечение компонент из Markdown
# ============================================================================

@dataclass
class MarkdownComponents:
    """Компоненты, извлеченные из Markdown-файла."""
    text_blocks: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    headers: List[Tuple[int, str]] = field(default_factory=list)  # (level, text)
    lists: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)  # пути к изображениям
    raw_content: str = ""

def extract_components(md_path: Path) -> MarkdownComponents:
    """Извлекает все компоненты из Markdown-файла."""
    content = md_path.read_text(encoding='utf-8')
    components = MarkdownComponents(raw_content=content)
    
    lines = content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Заголовки
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if header_match:
            level = len(header_match.group(1))
            text = header_match.group(2).strip()
            components.headers.append((level, text))
            i += 1
            continue
        
        # Таблицы (начинаются с |)
        if line.startswith('|'):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i])
                i += 1
            components.tables.append('\n'.join(table_lines))
            continue
        
        # Изображения
        image_match = re.findall(r'!\[.*?\]\((.*?)\)', line)
        for img_path in image_match:
            components.images.append(img_path)
        
        # Списки (маркированные или нумерованные)
        list_match = re.match(r'^(\s*)([-*+]|\d+\.)\s+(.+)$', line)
        if list_match:
            list_lines = []
            while i < len(lines) and (not lines[i].strip() or re.match(r'^(\s*)([-*+]|\d+\.)\s+', lines[i])):
                if lines[i].strip():
                    list_lines.append(lines[i])
                i += 1
            components.lists.append('\n'.join(list_lines))
            continue
        
        # Обычный текст (непустые строки, не заголовки)
        if line and not line.startswith('```'):
            # Собираем параграф
            para_lines = []
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(('#', '|', '```')):
                if not re.match(r'^!\[.*?\]\(.*?\)$', lines[i].strip()):
                    para_lines.append(lines[i].strip())
                i += 1
            if para_lines:
                components.text_blocks.append(' '.join(para_lines))
            continue
        
        i += 1
    
    return components

# ============================================================================
# Метрика для текста
# ============================================================================

def compute_text_score(pred_components: MarkdownComponents, 
                       gold_components: MarkdownComponents) -> float:
    """Вычисляет метрику для текста."""
    pred_text = ' '.join(pred_components.text_blocks)
    gold_text = ' '.join(gold_components.text_blocks)
    
    return levenshtein_similarity(pred_text, gold_text)

# ============================================================================
# Метрика для таблиц (упрощенный TEDS)
# ============================================================================

def parse_markdown_table(table_text: str) -> List[List[str]]:
    """Парсит Markdown таблицу в двумерный массив ячеек."""
    lines = table_text.strip().split('\n')
    rows = []
    
    for line in lines:
        if re.match(r'^[\s\|:\-]+$', line):
            continue  # Пропускаем строки-разделители
        
        # Разбиваем на ячейки
        cells = line.split('|')
        # Убираем пустые краевые ячейки
        if cells and cells[0].strip() == '':
            cells = cells[1:]
        if cells and cells[-1].strip() == '':
            cells = cells[:-1]
        
        rows.append([cell.strip() for cell in cells])
    
    return rows

def match_tables_greedy(pred_tables: List[str], gold_tables: List[str]) -> List[Tuple[int, int, float]]:
    """Жадное сопоставление таблиц."""
    if not pred_tables or not gold_tables:
        return []
    
    pred_parsed = [parse_markdown_table(t) for t in pred_tables]
    gold_parsed = [parse_markdown_table(t) for t in gold_tables]
    
    # Матрица сходства между таблицами
    similarities = []
    for i, pred in enumerate(pred_parsed):
        for j, gold in enumerate(gold_parsed):
            score = compute_table_similarity(pred, gold)
            similarities.append((score, i, j))
    
    # Сортируем по убыванию сходства
    similarities.sort(reverse=True, key=lambda x: x[0])
    
    matches = []
    used_pred = set()
    used_gold = set()
    
    for score, i, j in similarities:
        if i not in used_pred and j not in used_gold:
            matches.append((i, j, score))
            used_pred.add(i)
            used_gold.add(j)
    
    return matches

def compute_table_similarity(pred_table: List[List[str]], 
                             gold_table: List[List[str]]) -> float:
    """Вычисляет сходство двух таблиц."""
    if not pred_table or not gold_table:
        return 0.0
    
    # Выравниваем размеры таблиц
    max_rows = max(len(pred_table), len(gold_table))
    max_cols = max(
        max((len(row) for row in pred_table), default=0),
        max((len(row) for row in gold_table), default=0)
    )
    
    total_similarity = 0.0
    cell_count = 0
    
    for i in range(max_rows):
        pred_row = pred_table[i] if i < len(pred_table) else []
        gold_row = gold_table[i] if i < len(gold_table) else []
        
        for j in range(max_cols):
            pred_cell = pred_row[j] if j < len(pred_row) else ''
            gold_cell = gold_row[j] if j < len(gold_row) else ''
            
            if pred_cell or gold_cell:
                total_similarity += levenshtein_similarity(pred_cell, gold_cell)
                cell_count += 1
    
    if cell_count == 0:
        return 1.0
    
    return total_similarity / cell_count

def compute_tables_score(pred_components: MarkdownComponents,
                         gold_components: MarkdownComponents) -> float:
    """Вычисляет метрику для таблиц."""
    if not gold_components.tables:
        return 1.0 if not pred_components.tables else 0.0
    
    if not pred_components.tables:
        return 0.0
    
    matches = match_tables_greedy(pred_components.tables, gold_components.tables)
    
    if not matches:
        return 0.0
    
    # Среднее сходство по сопоставленным таблицам
    avg_similarity = sum(score for _, _, score in matches) / len(matches)
    
    # Штраф за разное количество таблиц
    count_penalty = min(len(pred_components.tables), len(gold_components.tables)) / \
                    max(len(pred_components.tables), len(gold_components.tables))
    
    return avg_similarity * count_penalty

# ============================================================================
# Метрика для структуры
# ============================================================================

def compute_headers_score(pred_headers: List[Tuple[int, str]], 
                          gold_headers: List[Tuple[int, str]]) -> float:
    """Вычисляет сходство заголовков."""
    if not gold_headers:
        return 1.0 if not pred_headers else 0.0
    
    if not pred_headers:
        return 0.0
    
    # Сравниваем заголовки по уровням
    pred_by_level = defaultdict(list)
    gold_by_level = defaultdict(list)
    
    for level, text in pred_headers:
        pred_by_level[level].append(text)
    for level, text in gold_headers:
        gold_by_level[level].append(text)
    
    scores = []
    for level in set(pred_by_level.keys()) | set(gold_by_level.keys()):
        pred_texts = pred_by_level.get(level, [])
        gold_texts = gold_by_level.get(level, [])
        
        # Жадное сопоставление заголовков одного уровня
        matches = []
        used_gold = set()
        
        for pred_text in pred_texts:
            best_score = 0.0
            best_idx = -1
            
            for j, gold_text in enumerate(gold_texts):
                if j not in used_gold:
                    score = levenshtein_similarity(pred_text, gold_text)
                    if score > best_score:
                        best_score = score
                        best_idx = j
            
            if best_idx >= 0:
                matches.append(best_score)
                used_gold.add(best_idx)
        
        if matches:
            scores.append(sum(matches) / len(matches))
        elif pred_texts or gold_texts:
            scores.append(0.0)
    
    if not scores:
        return 1.0
    
    return sum(scores) / len(scores)

def compute_lists_score(pred_lists: List[str], gold_lists: List[str]) -> float:
    """Вычисляет сходство списков."""
    if not gold_lists:
        return 1.0 if not pred_lists else 0.0
    
    if not pred_lists:
        return 0.0
    
    # Извлекаем элементы списков
    def extract_items(list_text: str) -> List[str]:
        items = []
        for line in list_text.split('\n'):
            match = re.match(r'^\s*[-*+]\s+(.+)$', line.strip())
            if not match:
                match = re.match(r'^\s*\d+\.\s+(.+)$', line.strip())
            if match:
                items.append(match.group(1).strip())
        return items
    
    pred_items = []
    for lst in pred_lists:
        pred_items.extend(extract_items(lst))
    
    gold_items = []
    for lst in gold_lists:
        gold_items.extend(extract_items(lst))
    
    if not gold_items:
        return 1.0
    
    # Жадное сопоставление элементов
    used_gold = set()
    matches = []
    
    for pred_item in pred_items:
        best_score = 0.0
        best_idx = -1
        
        for j, gold_item in enumerate(gold_items):
            if j not in used_gold:
                score = levenshtein_similarity(pred_item, gold_item)
                if score > best_score:
                    best_score = score
                    best_idx = j
        
        if best_idx >= 0:
            matches.append(best_score)
            used_gold.add(best_idx)
    
    if not matches:
        return 0.0
    
    # Учитываем полноту
    recall = len(matches) / len(gold_items)
    precision = len(matches) / len(pred_items) if pred_items else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    avg_similarity = sum(matches) / len(matches)
    
    return f1 * avg_similarity

def compute_structure_score(pred_components: MarkdownComponents,
                            gold_components: MarkdownComponents) -> float:
    """Вычисляет метрику для структуры."""
    headers_score = compute_headers_score(pred_components.headers, gold_components.headers)
    lists_score = compute_lists_score(pred_components.lists, gold_components.lists)
    
    # Веса внутри структуры: заголовки важнее списков
    return 0.6 * headers_score + 0.4 * lists_score

# ============================================================================
# Метрика для изображений
# ============================================================================

def compute_images_score(pred_components: MarkdownComponents,
                         gold_components: MarkdownComponents,
                         pred_dir: Path) -> float:
    """Вычисляет метрику для изображений."""
    # 1. Сравнение количества изображений
    pred_count = len(pred_components.images)
    gold_count = len(gold_components.images)
    
    if gold_count == 0:
        count_score = 1.0 if pred_count == 0 else 0.0
    else:
        count_score = min(pred_count, gold_count) / max(pred_count, gold_count)
    
    # 2. Проверка существования файлов изображений
    if pred_count == 0:
        existence_score = 0.0
    else:
        existing = 0
        for img_path in pred_components.images:
            # Проверяем, существует ли файл
            full_path = pred_dir / img_path
            if full_path.exists():
                existing += 1
        
        existence_score = existing / pred_count
    
    # Взвешенная сумма
    return 0.5 * count_score + 0.5 * existence_score

# ============================================================================
# Главная метрика
# ============================================================================

def compute_overall_score(pred_md_path: Path, gold_md_path: Path) -> Dict[str, float]:
    """Вычисляет все метрики для одного файла."""
    pred_components = extract_components(pred_md_path)
    gold_components = extract_components(gold_md_path)
    
    text_score = compute_text_score(pred_components, gold_components)
    tables_score = compute_tables_score(pred_components, gold_components)
    structure_score = compute_structure_score(pred_components, gold_components)
    images_score = compute_images_score(pred_components, gold_components, pred_md_path.parent)
    
    overall = (
        WEIGHTS['tables'] * tables_score +
        WEIGHTS['text'] * text_score +
        WEIGHTS['structure'] * structure_score +
        WEIGHTS['images'] * images_score
    )
    
    return {
        'overall': overall,
        'tables': tables_score,
        'text': text_score,
        'structure': structure_score,
        'images': images_score,
    }

def evaluate_all(pred_dir: Path, gold_dir: Path) -> Dict[str, any]:
    """Оценивает все файлы в директории."""
    pred_files = sorted(pred_dir.glob("*.md"))
    
    all_scores = []
    file_scores = {}
    
    for pred_path in pred_files:
        gold_path = gold_dir / pred_path.name
        
        if not gold_path.exists():
            print(f"Предупреждение: нет эталона для {pred_path.name}")
            continue
        
        scores = compute_overall_score(pred_path, gold_path)
        file_scores[pred_path.stem] = scores
        all_scores.append(scores['overall'])
    
    # Средние значения
    avg_scores = {
        'overall': sum(s['overall'] for s in file_scores.values()) / len(file_scores) if file_scores else 0.0,
        'tables': sum(s['tables'] for s in file_scores.values()) / len(file_scores) if file_scores else 0.0,
        'text': sum(s['text'] for s in file_scores.values()) / len(file_scores) if file_scores else 0.0,
        'structure': sum(s['structure'] for s in file_scores.values()) / len(file_scores) if file_scores else 0.0,
        'images': sum(s['images'] for s in file_scores.values()) / len(file_scores) if file_scores else 0.0,
    }
    
    return {
        'average': avg_scores,
        'per_file': file_scores,
        'weights': WEIGHTS,
    }

def main():
    parser = argparse.ArgumentParser(description="Оценка качества парсинга PDF в Markdown")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Директория с предсказаниями")
    parser.add_argument("--gold-dir", type=Path, required=True, help="Директория с эталонами")
    parser.add_argument("--output", type=Path, help="Файл для сохранения результатов (JSON)")
    args = parser.parse_args()
    
    print("Вычисление метрик...")
    results = evaluate_all(args.pred_dir, args.gold_dir)
    
    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 50)
    print(f"\nСредние метрики по {len(results['per_file'])} файлам:")
    print(f"  Общая:      {results['average']['overall']:.4f}")
    print(f"  Таблицы:    {results['average']['tables']:.4f} (вес: {WEIGHTS['tables']})")
    print(f"  Текст:      {results['average']['text']:.4f} (вес: {WEIGHTS['text']})")
    print(f"  Структура:  {results['average']['structure']:.4f} (вес: {WEIGHTS['structure']})")
    print(f"  Изображения: {results['average']['images']:.4f} (вес: {WEIGHTS['images']})")
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nРезультаты сохранены в: {args.output}")

if __name__ == "__main__":
    main()