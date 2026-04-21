"""Эвристики удаления повторяющихся «служебных» строк (колонтитулы, номера страниц)."""

from __future__ import annotations

import re
from typing import Dict

from baseline.table_cleanup import normalize_markdown_tables

_CYR_RE = re.compile(r"[А-Яа-яЁё]")
_LAT_RE = re.compile(r"[A-Za-z]")
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]{3,}")
_LAT_WORD_RE = re.compile(r"\b[A-Za-z]{4,}\b")
_WATERMARK_RE = re.compile(
    r"^(?:draft|черновик|copy|sample|confidential|preview)\b[^\S\r\n]*.*$",
    flags=re.IGNORECASE,
)
_WATERMARK_INLINE_RE = re.compile(
    r"\b(?:draft|черновик|confidential|preview)\b",
    flags=re.IGNORECASE,
)
_LAT_TO_CYR = str.maketrans(
    {
        "A": "А",
        "a": "а",
        "B": "В",
        "C": "С",
        "c": "с",
        "E": "Е",
        "e": "е",
        "H": "Н",
        "K": "К",
        "k": "к",
        "M": "М",
        "m": "м",
        "O": "О",
        "o": "о",
        "P": "Р",
        "p": "р",
        "T": "Т",
        "X": "Х",
        "x": "х",
        "Y": "У",
        "y": "у",
        # Частые OCR-подмены для "псевдо-кириллицы".
        "N": "П",
        "n": "п",
    }
)
_CONFUSABLE_LAT_CHARS = set("ABCEHKMOPTXYabcehkmoptxyNn")

def strip_repeating_noise_lines(text: str, *, min_repeats: int = 4) -> str:
    """
    Удалить короткие строки, встречающиеся на многих «страницах» (эвристика колонтитулов).

    Не трогает заголовки Markdown (#) и строки таблиц (|).
    """
    lines = text.split("\n")
    freq: Dict[str, int] = {}
    for line in lines:
        s = line.strip()
        if not s or len(s) > 120:
            continue
        if s.startswith("#") or s.startswith("|"):
            continue
        freq[s] = freq.get(s, 0) + 1

    drop = {s for s, c in freq.items() if c >= min_repeats}

    out: list[str] = []
    for line in lines:
        s = line.strip()
        if s in drop:
            continue
        out.append(line)

    # сжать множественные пустые строки
    collapsed: list[str] = []
    empty_run = 0
    for line in out:
        if not line.strip():
            empty_run += 1
            if empty_run <= 2:
                collapsed.append(line)
        else:
            empty_run = 0
            collapsed.append(line)

    text2 = "\n".join(collapsed).strip()
    return text2 + ("\n" if text2 else "")


def strip_simple_page_markers(text: str) -> str:
    """Убрать отдельные строки вида «стр. N», «Page N», чистые номера страниц."""
    patterns = (
        re.compile(r"^(стр\.|страница|page)\s*\d+\s*$", re.IGNORECASE),
        re.compile(r"^\[\s*\d+\s*\]$"),
        re.compile(r"^\d+\s*/\s*\d+$"),
    )
    lines = text.split("\n")
    kept: list[str] = []
    for line in lines:
        s = line.strip()
        if any(p.match(s) for p in patterns):
            continue
        kept.append(line)
    return "\n".join(kept).strip() + "\n"


def strip_watermarks(text: str) -> str:
    """
    Удалить водяные знаки/служебные метки, которые часто попадают в текст и таблицы.
    Делаем максимально консервативно: удаляем целые строки, где это почти наверняка watermark,
    и вырезаем inline-токены только если строка короткая.
    """
    out: list[str] = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            out.append(line)
            continue
        # Не трогаем заголовки (на всякий случай) и картинки.
        if s.startswith("#") or s.startswith("!["):
            out.append(line)
            continue
        if _WATERMARK_RE.match(s) and len(s) <= 80:
            continue
        is_table_row = s.startswith("|")
        # Inline:
        # - в обычном тексте — только в коротких строках;
        # - в таблицах — можно смелее, watermark почти никогда не является полезным значением ячейки.
        if _WATERMARK_INLINE_RE.search(s) and (is_table_row or len(s) <= 60):
            cleaned = _WATERMARK_INLINE_RE.sub("", line)
            if is_table_row:
                # В таблицах сохраняем разделители, но убираем лишние пробелы.
                cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
                cleaned = re.sub(r"\|\s+\|", "|  |", cleaned)
                cleaned = cleaned.rstrip()
            else:
                cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
            if not cleaned.strip():
                continue
            out.append(cleaned)
            continue
        out.append(line)
    return "\n".join(out)


def _repair_mixed_script_word(word: str) -> str:
    """
    Исправить смешанную латиницу/кириллицу внутри одного слова:
    «Кoрoбка» -> «Коробка», «Рaздeл» -> «Раздел».
    """
    if not _CYR_RE.search(word) or not _LAT_RE.search(word):
        return word
    repaired = word.translate(_LAT_TO_CYR)
    # Применяем только если латиницы стало меньше.
    if len(_LAT_RE.findall(repaired)) < len(_LAT_RE.findall(word)):
        return repaired
    return word


def repair_mixed_script_text(text: str) -> str:
    """Починить «битые» слова со смешением латиницы и кириллицы."""
    lines = text.split("\n")
    out: list[str] = []
    for line in lines:
        # В таблицах это особенно заметно, но применяем и к обычному тексту.
        repaired = _WORD_RE.sub(lambda m: _repair_mixed_script_word(m.group(0)), line)
        out.append(repaired)
    return "\n".join(out)


def _repair_latin_word_in_cyr_context(word: str, *, line_has_cyr: bool) -> str:
    """
    Исправить латинское слово, если оно с высокой вероятностью является
    псевдо-кириллицей (например, "FnaBa" из OCR). Применяем только в строках
    с кириллицей, чтобы минимизировать порчу реального английского текста.
    """
    if not line_has_cyr:
        return word
    if any(ch not in _CONFUSABLE_LAT_CHARS for ch in word):
        return word
    repaired = word.translate(_LAT_TO_CYR)
    # Подстраховка: если после замены не появилось кириллицы — не меняем.
    if _CYR_RE.search(repaired):
        return repaired
    return word


def repair_confusable_latin_in_cyr_lines(text: str) -> str:
    """
    Починить полностью латинские токены, собранные OCR из похожих символов,
    но только в строках с кириллицей (заголовки/таблицы русского документа).
    """
    out: list[str] = []
    for line in text.split("\n"):
        has_cyr = bool(_CYR_RE.search(line))
        repaired = _LAT_WORD_RE.sub(
            lambda m: _repair_latin_word_in_cyr_context(m.group(0), line_has_cyr=has_cyr),
            line,
        )
        out.append(repaired)
    return "\n".join(out)


def postprocess_markdown(text: str, *, strip_noise: bool = True) -> str:
    """Пайплайн постобработки после Docling."""
    text = normalize_markdown_tables(text)
    text = strip_watermarks(text)
    # Склейка переносов в обычном тексте (в таблицах это делается на уровне ячейки).
    # Важно делать до repair_*: иначе "остав-\nить" превращается в "остав ить".
    text = text.replace("\u00ad", "")
    text = re.sub(
        r"([A-Za-zА-Яа-яЁё])[\-\u2010\u2011\u2012\u2013\u2212]\s*\n\s*([A-Za-zА-Яа-яЁё])",
        r"\1\2",
        text,
    )
    text = re.sub(
        r"(?<!\s)([A-Za-zА-Яа-яЁё])\n([A-Za-zА-Яа-яЁё])(?!\s)",
        r"\1\2",
        text,
    )
    text = repair_mixed_script_text(text)
    text = repair_confusable_latin_in_cyr_lines(text)
    if strip_noise:
        text = strip_simple_page_markers(text)
        text = strip_repeating_noise_lines(text)
    return text
