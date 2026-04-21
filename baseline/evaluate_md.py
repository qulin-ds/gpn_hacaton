"""Проверка Markdown-результатов: sanity-check и опционально сравнение с ground truth."""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

_CYR_RE = re.compile(r"[А-Яа-яЁё]")
_LAT_RE = re.compile(r"[A-Za-z]")
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]{3,}")


def _count_broken_cyr_words(content: str) -> tuple[int, int]:
    """
    Вернуть (mixed_script_words, pseudo_cyr_latin_words):
    - mixed_script_words: слова со смешанной кириллицей/латиницей;
    - pseudo_cyr_latin_words: полностью латинские слова в строках с кириллицей
      (частая OCR-псевдо-кириллица).
    """
    mixed = 0
    pseudo = 0
    for line in content.split("\n"):
        has_cyr_line = bool(_CYR_RE.search(line))
        words = _WORD_RE.findall(line)
        for w in words:
            has_cyr = bool(_CYR_RE.search(w))
            has_lat = bool(_LAT_RE.search(w))
            if has_cyr and has_lat:
                mixed += 1
            elif has_cyr_line and has_lat and not has_cyr:
                pseudo += 1
    return mixed, pseudo


def _count_broken_cyr_words_in_tables(content: str) -> tuple[int, int]:
    """То же, но только по строкам markdown-таблиц (начинаются с '|')."""
    table_lines = [ln for ln in content.split("\n") if ln.strip().startswith("|")]
    if not table_lines:
        return 0, 0
    return _count_broken_cyr_words("\n".join(table_lines))


def analyze_md_file(md_path: Path) -> dict[str, Any]:
    """Анализ одного MD файла (ссылки на картинки, таблицы, «мусорные» строки)."""
    try:
        content = md_path.read_text(encoding="utf-8")
    except Exception as e:
        return {"filename": md_path.name, "error": str(e)}

    lines = content.split("\n")

    stats: dict[str, Any] = {
        "filename": md_path.name,
        "size_kb": len(content.encode("utf-8")) / 1024,
        "char_count": len(content),
    }

    headers = re.findall(r"^(#{1,6})\s+(.+)$", content, re.MULTILINE)
    stats["headers_count"] = len(headers)

    table_count = len(re.findall(r"\|[\s\-:|]+\|", content))
    stats["tables_count"] = table_count

    img_links = re.findall(r"!\[.*?\]\((.*?)\)", content)
    stats["images_linked"] = len(img_links)

    existing_imgs = 0
    missing_imgs = 0
    wrong_name = 0
    stem = md_path.stem
    m = re.match(r"document_(\d+)$", stem)
    doc_id = int(m.group(1)) if m else None
    img_name_re = re.compile(rf"^images/doc_{doc_id}_image_\d+\.png$") if doc_id is not None else None

    for link in img_links:
        clean_link = link.split("#")[0].strip()
        if not clean_link:
            continue
        img_path = md_path.parent / clean_link
        if img_path.exists():
            existing_imgs += 1
        else:
            missing_imgs += 1
        if img_name_re is not None and clean_link and not img_name_re.match(clean_link.replace("\\", "/")):
            wrong_name += 1

    stats["images_existing"] = existing_imgs
    stats["images_missing"] = missing_imgs
    stats["images_suspicious_name"] = wrong_name

    long_lines_no_space = sum(1 for line in lines if len(line) > 150 and " " not in line)
    stats["long_garbage_lines"] = long_lines_no_space
    mixed_all, pseudo_all = _count_broken_cyr_words(content)
    mixed_tbl, pseudo_tbl = _count_broken_cyr_words_in_tables(content)
    stats["broken_cyr_mixed_words"] = mixed_all
    stats["broken_cyr_pseudo_latin_words"] = pseudo_all
    stats["broken_cyr_mixed_words_tables"] = mixed_tbl
    stats["broken_cyr_pseudo_latin_words_tables"] = pseudo_tbl

    return stats


def _compare_to_reference(pred_path: Path, ref_path: Path) -> float | None:
    if not ref_path.is_file():
        return None
    pred = pred_path.read_text(encoding="utf-8")
    ref = ref_path.read_text(encoding="utf-8")
    return float(fuzz.ratio(pred, ref))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Проверка каталога с .md: ссылки на изображения и опционально сравнение с эталоном.",
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Каталог с предсказанными .md")
    parser.add_argument("--max-files", "-n", type=int, default=None, help="Ограничить число файлов")
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=None,
        help="Каталог с эталонными .md (те же имена файлов, что и в input-dir)",
    )
    parser.add_argument(
        "--strict-cyr",
        action="store_true",
        help="Показать TOP проблемных файлов по OCR-артефактам кириллицы в таблицах.",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Ошибка: папка {args.input_dir} не найдена.")
        return

    md_files = sorted(args.input_dir.glob("*.md"))
    if not md_files:
        print("Нет .md файлов.")
        return

    if args.max_files is not None:
        md_files = md_files[: args.max_files]
        print(f"Анализ первых {len(md_files)} файлов...\n")
    else:
        print(f"Анализ всех {len(md_files)} файлов...\n")

    anomalies: list[dict[str, Any]] = []
    cyr_ranking: list[tuple[str, int, int, int]] = []
    total_miss = 0
    total_imgs = 0
    total_broken_cyr = 0
    total_broken_cyr_tbl = 0
    ratios: list[float] = []

    gt_dir = args.ground_truth_dir
    for idx, md_path in enumerate(md_files):
        stats = analyze_md_file(md_path)
        if "error" in stats:
            print(stats)
            continue
        total_imgs += stats["images_linked"]
        total_miss += stats["images_missing"]
        total_broken_cyr += stats.get("broken_cyr_mixed_words", 0) + stats.get("broken_cyr_pseudo_latin_words", 0)
        total_broken_cyr_tbl += stats.get("broken_cyr_mixed_words_tables", 0) + stats.get(
            "broken_cyr_pseudo_latin_words_tables",
            0,
        )
        cyr_tbl_mixed = stats.get("broken_cyr_mixed_words_tables", 0)
        cyr_tbl_pseudo = stats.get("broken_cyr_pseudo_latin_words_tables", 0)
        cyr_ranking.append((stats["filename"], cyr_tbl_mixed + cyr_tbl_pseudo, cyr_tbl_mixed, cyr_tbl_pseudo))

        ratio_val: float | None = None
        if gt_dir is not None:
            ref = gt_dir / md_path.name
            ratio_val = _compare_to_reference(md_path, ref)
            if ratio_val is not None:
                ratios.append(ratio_val)

        reasons: list[str] = []
        if stats["images_missing"] > 0:
            reasons.append(f"Потеряно {stats['images_missing']} картинок")
        if stats.get("images_suspicious_name", 0) > 0:
            reasons.append(f"Подозрительные имена ссылок: {stats['images_suspicious_name']}")
        if stats["long_garbage_lines"] > 3:
            reasons.append(f"{stats['long_garbage_lines']} длинных строк без пробелов")
        if stats.get("broken_cyr_mixed_words_tables", 0) + stats.get("broken_cyr_pseudo_latin_words_tables", 0) > 6:
            reasons.append(
                "Много OCR-артефактов в таблицах: "
                f"{stats.get('broken_cyr_mixed_words_tables', 0)} mixed + "
                f"{stats.get('broken_cyr_pseudo_latin_words_tables', 0)} pseudo-latin",
            )

        if reasons:
            anomalies.append({"file": stats["filename"], "reasons": reasons})

        if idx < 10:
            extra = f" | fuzz: {ratio_val:.1f}" if ratio_val is not None else ""
            print(
                f"{stats['filename']:<25} | Карт: {stats['images_linked']} "
                f"(Miss: {stats['images_missing']}) | Табл: {stats['tables_count']} | "
                f"OCR-cyr(tbl): {stats.get('broken_cyr_mixed_words_tables', 0) + stats.get('broken_cyr_pseudo_latin_words_tables', 0)}"
                f"{extra}",
            )

    print("\n" + "=" * 70)
    print(f"Всего ссылок на картинки: {total_imgs}, битых: {total_miss}")
    print(f"Оценка OCR-артефактов (все строки): {total_broken_cyr}")
    print(f"Оценка OCR-артефактов (только таблицы): {total_broken_cyr_tbl}")
    if args.strict_cyr:
        ranked = sorted(cyr_ranking, key=lambda x: x[1], reverse=True)
        top = [x for x in ranked if x[1] > 0][:10]
        print("\nTOP проблемных файлов по OCR-артефактам в таблицах:")
        if top:
            for fname, score, mixed, pseudo in top:
                print(f"  - {fname}: score={score} (mixed={mixed}, pseudo-latin={pseudo})")
        else:
            print("  - проблемных файлов не найдено")
    if ratios:
        avg = sum(ratios) / len(ratios)
        print(f"Среднее сходство с эталоном (fuzz.ratio, {len(ratios)} файлов): {avg:.2f}")

    if anomalies:
        print(f"\nНайдено {len(anomalies)} файлов с замечаниями:")
        for a in anomalies[:10]:
            print(f"  - {a['file']}: {', '.join(a['reasons'])}")
    else:
        print("\nКритичных проблем по ссылкам/мусору не обнаружено.")


def build_submission_cli() -> None:
    """CLI: py -m baseline.evaluate_md не используется; вызывается из entrypoint."""
    p = argparse.ArgumentParser(description="Собрать submission.zip")
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("submission.zip"))
    args = p.parse_args()
    build_submission_zip(args.source_dir, args.output)
    print(f"Готово: {args.output.resolve()}")


def build_submission_zip(source_dir: Path, zip_path: Path) -> None:
    """
    Собрать submission.zip: только document_*.md и images/*.png (без прочих файлов).
    source_dir — каталог, где лежат .md и подкаталог images/.
    """
    md_files = sorted(source_dir.glob("document_*.md"))
    img_dir = source_dir / "images"
    if not md_files:
        raise FileNotFoundError(f"Нет document_*.md в {source_dir}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for md in md_files:
            zf.write(md, arcname=md.name)
        if img_dir.is_dir():
            for png in sorted(img_dir.glob("*.png")):
                zf.write(png, arcname=f"images/{png.name}")


if __name__ == "__main__":
    main()
