"""Utilities for lightweight Markdown table normalization."""

from __future__ import annotations

import re


_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_SEP_CELL_RE = re.compile(r"^\s*:?-{3,}:?\s*$")
# Unicode «тире», которые иногда попадают в строку-разделитель из PDF.
_UNICODE_DASHES_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]+")
# Строка вида "| весь текст без внутренних колонок |" — не Markdown-таблица.
_SINGLE_WRAPPED_ROW_RE = re.compile(r"^\s*\|([^|]*)\|\s*$")
_IMG_LINE_RE = re.compile(r"^\s*!\[.*?\]\(.*?\)\s*$")
_CAPTION_LINE_RE = re.compile(r"^\s*(рис\.|fig\.|figure)\s*\d+[\.:)]", flags=re.IGNORECASE)
_HLINE_DASH_ROW_RE = re.compile(r"^\s*\|\s*-\s*(\|\s*-\s*)+\|\s*$")


def _split_table_block(lines: list[str], start: int) -> tuple[list[str], int]:
    """Read contiguous markdown-table rows starting at start."""
    block: list[str] = []
    i = start
    n = len(lines)
    while i < n and _TABLE_ROW_RE.match(lines[i]):
        block.append(lines[i])
        i += 1
    return block, i


def _split_cells(row: str) -> list[str]:
    core = row.strip().strip("|")
    return [c.strip() for c in core.split("|")]


def _mk_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _normalize_separator_token(cell: str) -> str:
    """Привести содержимое ячейки-разделителя к ASCII-дефисам для проверки и вывода."""
    s = _UNICODE_DASHES_RE.sub("-", cell.strip())
    return s


def _is_separator_row(row: str) -> bool:
    cells = _split_cells(row)
    if not cells:
        return False
    for c in cells:
        norm = _normalize_separator_token(c or "---")
        if not _SEP_CELL_RE.match(norm):
            return False
    return True


def _separator_cells_normalized(row: str, col_count: int) -> list[str]:
    """Ячейки разделителя: только валидные `---` / `:---` и т.д., длина = col_count."""
    raw = _split_cells(row)
    raw = (raw + ["---"] * col_count)[:col_count]
    out: list[str] = []
    for c in raw:
        norm = _normalize_separator_token(c or "---")
        if _SEP_CELL_RE.match(norm):
            out.append(norm.strip())
        else:
            out.append("---")
    return out


def _clean_cell(text: str) -> str:
    """Сжать пробелы и переносы внутри ячейки (после tabulate / переносов из PDF)."""
    s = text.replace("\r\n", "\n").replace("\r", "\n")

    # 1) Удалить мягкие дефисы, часто попадающие из PDF/OCR.
    s = s.replace("\u00ad", "")

    # 2) Склеить дефисные переносы: "остав-\nить" -> "оставить"
    #    Включаем и unicode-дефисы/минусы, которые могут появляться в PDF.
    s = re.sub(
        r"([A-Za-zА-Яа-яЁё])[\-\u2010\u2011\u2012\u2013\u2212]\s*\n\s*([A-Za-zА-Яа-яЁё])",
        r"\1\2",
        s,
    )

    # 3) Склеить перенос внутри слова без дефиса: "остав\nить" -> "оставить".
    #    Делаем это только когда перенос окружён буквами без пробелов, чтобы
    #    не склеить "слово\nслово" (там должен быть пробел).
    s = re.sub(
        r"(?<!\s)([A-Za-zА-Яа-яЁё])\n([A-Za-zА-Яа-яЁё])(?!\s)",
        r"\1\2",
        s,
    )

    # 4) Остальные переносы считаем обычными пробелами (перенос строки в ячейке).
    s = s.replace("\n", " ")

    # 5) Сжать пробелы.
    s = re.sub(r"[ \t\f\v]+", " ", s)
    # 6) Водяные знаки в ячейках (часто попадают внутрь таблицы).
    s = re.sub(r"\b(?:draft|черновик|confidential|preview)\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[ \t\f\v]{2,}", " ", s)
    return s.strip()


def _dedupe_row_cells_for_colspan(cells: list[str]) -> list[str]:
    """
    Эвристика для "сложных" таблиц, где объединённые ячейки (colspan) в Markdown
    часто выглядят как дублирование одного и того же текста в соседних колонках.

    Для TEDS обычно лучше оставить текст в первой ячейке серии и обнулить повторы.
    """
    # Пользовательский режим: сохраняем дублирование "как есть".
    return cells


def _looks_like_same_table_header(a: str, b: str) -> bool:
    """Консервативная проверка: два заголовка совпадают после чистки ячеек."""
    ca = [_clean_cell(x) for x in _split_cells(a)]
    cb = [_clean_cell(x) for x in _split_cells(b)]
    if not ca or not cb:
        return False
    if len(ca) != len(cb):
        return False
    # Полное совпадение слишком строго, но чаще всего это и нужно.
    return ca == cb


def merge_split_tables(text: str) -> str:
    """
    Склеить таблицы, которые Docling разрывает вставкой изображения/подписи между
    двумя блоками одной и той же таблицы.

    Сценарий:
      | H1 | H2 |
      | ---| ---|
      | ...|
      ![Image](...)
      | H1 | H2 |
      | ---| ---|
      | ...|

    => оставить одну шапку, добавить строки второй части.
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        if not _TABLE_ROW_RE.match(lines[i]):
            out.append(lines[i])
            i += 1
            continue

        block1, j = _split_table_block(lines, i)
        if not block1:
            out.append(lines[i])
            i += 1
            continue

        # Пропускаем прослойку: пустые строки, подписи, картинки.
        k = j
        gap: list[str] = []
        while k < n:
            ln = lines[k]
            if not ln.strip() or _IMG_LINE_RE.match(ln) or _CAPTION_LINE_RE.match(ln.strip()):
                gap.append(ln)
                k += 1
                continue
            break

        if k < n and _TABLE_ROW_RE.match(lines[k]):
            block2, k2 = _split_table_block(lines, k)
            if block2 and len(block1) >= 2 and len(block2) >= 2:
                # Склеиваем только если заголовки совпадают (минимизируем ложные склейки).
                if _looks_like_same_table_header(block1[0], block2[0]):
                    # Собираем: block1 целиком + body block2 (без header+separator).
                    out.extend(block1)
                    # Вставку (картинку/подпись) сохраняем на месте — иначе ломаем текстовый порядок.
                    out.extend(gap)
                    body2 = block2[2:] if _is_separator_row(block2[1]) else block2[1:]
                    out.extend(body2)
                    i = k2
                    continue

        # Не склеили — просто вывести block1 и продолжить.
        out.extend(block1)
        i = j
    return "\n".join(out)


def _is_fake_pipe_wrapped_prose_block(block: list[str]) -> bool:
    """
    Docling иногда даёт один длинный абзац в одной паре |...| без внутренних колонок.
    Раньше сюда же попадали многострочные одноколоночные «таблицы» TableFormer — их разворачивали
    в простыню текста (баг вроде document_003). Разворачиваем только одиночную длинную строку.
    """
    if not block:
        return False
    for row in block:
        m = _SINGLE_WRAPPED_ROW_RE.match(row)
        if not m:
            return False
        inner = m.group(1).strip()
        if "|" in inner:
            return False
    if len(block) == 1:
        inner = _SINGLE_WRAPPED_ROW_RE.match(block[0])
        if inner and len(inner.group(1).strip()) >= 120:
            return True
    return False


def _row_col_count(row: str) -> int:
    return len(_split_cells(row))


def _unwrap_fake_pipe_block(block: list[str]) -> list[str]:
    out: list[str] = []
    for row in block:
        m = _SINGLE_WRAPPED_ROW_RE.match(row)
        if m:
            out.append(m.group(1).strip())
    return out


def normalize_markdown_tables(text: str) -> str:
    """
    Normalize Markdown tables (минимально инвазивно):
    - гарантировать одну строку-разделитель после заголовка;
    - кол-во колонок берём по заголовку (а не max по блоку), чтобы шумные строки
      с лишними '|' не раздували таблицу;
    - склеить блок, разорванный одной пустой строкой;
    - остальные строки оставить максимально близко к исходнику.
    """
    # 1) Пробуем склеить разорванные таблицы до нормализации.
    text = merge_split_tables(text)
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        if not _TABLE_ROW_RE.match(line):
            out.append(line)
            i += 1
            continue

        block: list[str] = []
        ref_cols: int | None = None
        while i < n:
            if _TABLE_ROW_RE.match(lines[i]):
                if ref_cols is None:
                    ref_cols = _row_col_count(lines[i])
                block.append(lines[i])
                i += 1
                continue
            # Docling иногда вставляет пустую строку между строками одной таблицы — без склейки GFM ломается.
            if (
                not lines[i].strip()
                and i + 1 < n
                and _TABLE_ROW_RE.match(lines[i + 1])
                and ref_cols is not None
                and ref_cols >= 2
            ):
                nxt = lines[i + 1]
                nc = _row_col_count(nxt)
                if nc == ref_cols or abs(nc - ref_cols) <= 1:
                    if _is_separator_row(nxt) and any(_is_separator_row(r) for r in block):
                        break
                    i += 1
                    continue
            break

        if not block:
            continue

        if _is_fake_pipe_wrapped_prose_block(block):
            out.extend(_unwrap_fake_pipe_block(block))
            continue

        # Кол-во колонок: по заголовку (как правило совпадает с эталоном).
        header_raw = _split_cells(block[0])
        col_count = max(1, len(header_raw))
        header_cells = [_clean_cell(c) for c in header_raw]
        header_cells = (header_cells + [""] * col_count)[:col_count]

        normalized: list[str] = []
        normalized.append(_mk_row(header_cells))

        # Ровно один separator после заголовка.
        body_idx = 1
        if len(block) >= 2 and _is_separator_row(block[1]):
            normalized.append(_mk_row(_separator_cells_normalized(block[1], col_count)))
            body_idx = 2
        else:
            normalized.append(_mk_row(["---"] * col_count))

        for row in block[body_idx:]:
            # Дополнительные separator-строки внутри блока чаще вредят совпадению с эталоном.
            if _is_separator_row(row):
                continue
            # Типичная мусорная строка-заполнитель: "| - | - | - |"
            if _HLINE_DASH_ROW_RE.match(row):
                continue
            cells = [_clean_cell(c) for c in _split_cells(row)]
            # Делаем ряд совместимым с заголовком: обрезать/дополнить до col_count.
            if len(cells) >= col_count:
                cells = cells[:col_count]
            else:
                cells = cells + [""] * (col_count - len(cells))
            cells = _dedupe_row_cells_for_colspan(cells)
            normalized.append(_mk_row(cells))

        out.extend(normalized)

    return "\n".join(out)
