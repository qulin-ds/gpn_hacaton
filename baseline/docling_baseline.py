from __future__ import annotations

import argparse
import gc
import importlib
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

_CYR_RE = re.compile(r"[А-Яа-яЁё]")
_LAT_RE = re.compile(r"[A-Za-z]")


def _apply_device_from_argv() -> None:
    """Docling читает устройство из ``AcceleratorOptions`` / ``DOCLING_DEVICE``."""
    for i, arg in enumerate(sys.argv):
        if arg == "--device" and i + 1 < len(sys.argv):
            val = sys.argv[i + 1]
            if val != "auto":
                os.environ["DOCLING_DEVICE"] = val
            return
        if arg.startswith("--device="):
            val = arg.split("=", 1)[1]
            if val != "auto":
                os.environ["DOCLING_DEVICE"] = val
            return


_apply_device_from_argv()


def _patch_cv2_set_num_threads() -> None:
    """TableFormer (docling-ibm-models) вызывает ``cv2.setNumThreads``; у подменного cv2 атрибута нет."""
    try:
        import cv2
    except ImportError:
        return
    if not hasattr(cv2, "setNumThreads"):
        cv2.setNumThreads = lambda _nthreads: None  # type: ignore[method-assign]


_patch_cv2_set_num_threads()

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    TableFormerMode,
    TableStructureOptions,
    ThreadedPdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.base import ImageRefMode

from baseline.text_cleanup import postprocess_markdown

_IMG_LINK_RE = re.compile(
    r"images[\\/](image_\d+_[a-f0-9]+\.png)",
    flags=re.IGNORECASE,
)


def _clear_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


def _doc_num_from_stem(stem: str) -> int | None:
    """Как в evaluation: ``document_051`` -> 51."""
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _normalize_image_names(
    markdown: str,
    work_images_dir: Path,
    out_images_dir: Path,
    doc_num: int,
) -> str:
    """Переименовать ``image_*_*.png`` в ``doc_<n>_image_<k>.png`` и обновить ссылки."""
    out_images_dir.mkdir(parents=True, exist_ok=True)
    old_to_new: dict[str, str] = {}
    order = 1
    for m in _IMG_LINK_RE.finditer(markdown):
        old_name = m.group(1)
        if old_name in old_to_new:
            continue
        src = work_images_dir / old_name
        if not src.is_file():
            continue
        new_name = f"doc_{doc_num}_image_{order}.png"
        old_to_new[old_name] = new_name
        shutil.move(str(src), str(out_images_dir / new_name))
        order += 1

    out = markdown
    for old_name, new_name in sorted(old_to_new.items(), key=lambda kv: len(kv[0]), reverse=True):
        out = out.replace(f"images/{old_name}", f"images/{new_name}")
        out = out.replace(f"images\\{old_name}", f"images/{new_name}")
        out = out.replace(f"/images/{old_name}", f"/images/{new_name}")

    # Docling может писать абсолютные пути до artifacts_dir (особенно на Windows).
    # Оставляем только относительный путь "images/<name>".
    out = re.sub(
        r"[A-Za-z]:[^)\n]*?/images/(doc_\d+_image_\d+\.png)",
        r"images/\1",
        out,
    )
    return out


def _build_converter(
    do_ocr: bool,
    no_table_structure: bool,
    full_quality: bool,
    ocr_languages: list[str],
) -> DocumentConverter:
    if full_quality:
        images_scale = 1.0
        table_opts = TableStructureOptions(mode=TableFormerMode.ACCURATE)
    else:
        images_scale = 0.88
        table_opts = TableStructureOptions(mode=TableFormerMode.FAST)

    pipeline_options = ThreadedPdfPipelineOptions(
        do_ocr=do_ocr,
        do_table_structure=not no_table_structure,
        generate_picture_images=True,
        images_scale=images_scale,
        table_structure_options=table_opts,
        accelerator_options=AcceleratorOptions(),
    ).model_copy(deep=True)
    # В разных версиях Docling имя атрибута может отличаться:
    # пробуем задать языки OCR максимально совместимо.
    if do_ocr:
        for field_name in ("ocr_options", "ocr", "rapid_ocr_options"):
            ocr_obj = getattr(pipeline_options, field_name, None)
            if ocr_obj is None:
                continue
            for lang_field in ("lang", "langs", "languages"):
                if hasattr(ocr_obj, lang_field):
                    try:
                        setattr(ocr_obj, lang_field, ocr_languages)
                    except Exception:
                        pass
    # Дополнительный путь: часть версий читает языки из переменной окружения.
    os.environ["DOCLING_OCR_LANGUAGES"] = ",".join(ocr_languages)
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        },
    )


def _pdf_has_text_layer(pdf_path: Path, *, min_chars: int = 32) -> bool:
    """
    Эвристика: есть ли в PDF текстовый слой.
    Если есть, лучше выключать OCR, чтобы не вносить OCR-ошибки в кириллицу.
    """
    def _looks_like_real_text(s: str) -> bool:
        t = (s or "").strip()
        if len(t) < min_chars:
            return False
        # Частый кейс “слой есть, но мусор”: много непечатаемых/заменителей.
        printable = sum(1 for ch in t if ch.isprintable())
        if printable / max(1, len(t)) < 0.85:
            return False
        # Хоть немного букв (кириллица/латиница) — иначе это может быть «пустой» слой.
        if not (_CYR_RE.search(t) or _LAT_RE.search(t)):
            return False
        return True

    try:
        pypdf = importlib.import_module("pypdf")
        PdfReader = getattr(pypdf, "PdfReader")
    except Exception:
        # Если pypdf не установлен, безопасно считаем, что слой неизвестен.
        return False
    try:
        reader = PdfReader(str(pdf_path))
        sample_pages = min(5, len(reader.pages))
        total = 0
        buf: list[str] = []
        for i in range(sample_pages):
            text = reader.pages[i].extract_text() or ""
            t = text.strip()
            total += len(t)
            if t:
                buf.append(t)
            if total >= min_chars:
                return _looks_like_real_text("\n".join(buf))
    except Exception:
        return False
    return False


def _md_looks_empty(md_path: Path, *, min_chars: int = 250) -> bool:
    """
    Проверка результата: если Markdown получился «почти пустым», вероятно,
    у PDF был проблемный текстовый слой и надо перепрогнать с OCR.
    """
    try:
        txt = md_path.read_text(encoding="utf-8")
    except Exception:
        return True
    t = txt.strip()
    if len(t) < min_chars:
        return True
    letters = sum(1 for ch in t if ch.isalpha())
    if letters / max(1, len(t)) < 0.08:
        return True
    return False


def convert_pdf(
    pdf_path: Path,
    output_dir: Path,
    converter: DocumentConverter,
    *,
    text_cleanup: bool = True,
) -> None:
    stem = pdf_path.stem
    doc_num = _doc_num_from_stem(stem)
    if doc_num is None:
        doc_num = 1

    result = converter.convert(str(pdf_path))
    doc = result.document

    with tempfile.TemporaryDirectory(prefix=f"docling_{stem}_") as tmp:
        work = Path(tmp)
        md_work = work / f"{stem}.md"
        # compact_tables убирает выравнивание пробелами от tabulate (широкие/«сложные» таблицы иначе раздуваются).
        _save_kw: dict = {
            "filename": md_work,
            "artifacts_dir": work / "images",
            "image_mode": ImageRefMode.REFERENCED,
            "compact_tables": True,
        }
        try:
            doc.save_as_markdown(**_save_kw)
        except TypeError:
            _save_kw.pop("compact_tables", None)
            doc.save_as_markdown(**_save_kw)
        text = md_work.read_text(encoding="utf-8")
        # Нормализуем слеши перед поиском изображений/ссылок.
        text = text.replace("\\", "/")
        if text_cleanup:
            text = postprocess_markdown(text)
        text = _normalize_image_names(
            text,
            work_images_dir=work / "images",
            out_images_dir=output_dir / "images",
            doc_num=doc_num,
        )
        out_md = output_dir / f"{stem}.md"
        out_md.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline (Docling): PDF -> Markdown")
    parser.add_argument("--input-dir", type=Path, required=True, help="Директория с PDF")
    parser.add_argument("--output-dir", type=Path, required=True, help="Каталог результатов")
    parser.add_argument(
        "--only-docs",
        type=str,
        default=None,
        help="Обработать только указанные документы по номерам: например '7,21,24'. "
        "Номера берутся из имени document_XXX.pdf.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Ограничить число файлов (для отладки)",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Отключить OCR (быстрее на PDF с текстовым слоем; сканы будут хуже)",
    )
    parser.add_argument(
        "--ocr-mode",
        choices=("auto", "on", "off"),
        default="auto",
        help="Режим OCR: auto (по текстовому слою PDF), on (всегда OCR), off (всегда без OCR).",
    )
    parser.add_argument(
        "--ocr-languages",
        type=str,
        default="ru,en",
        help="Языки OCR через запятую, например: ru,en",
    )
    parser.add_argument(
        "--no-table-structure",
        action="store_true",
        help="Отключить TableFormer (быстрее; таблицы часто превращаются в текст в одной паре |...| — для хакатона не рекомендуется).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Быстрый режим: то же что --no-ocr --no-table-structure (хуже таблицы и сканы).",
    )
    parser.add_argument(
        "--full-quality",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Полное качество: images_scale=1.0 и TableFormer ACCURATE (по умолчанию вкл.; "
        "--no-full-quality — быстрее, хуже сложные таблицы).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Устройство для моделей Docling (auto = по умолчанию). "
        "Задаётся до импорта через DOCLING_DEVICE.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Не обрабатывать PDF, если в output уже есть одноимённый .md (продолжение после обрыва).",
    )
    parser.add_argument(
        "--no-text-cleanup",
        action="store_true",
        help="Отключить эвристическую очистку повторяющихся колонтитулов/маркеров страниц.",
    )
    args = parser.parse_args()
    if args.fast:
        args.no_ocr = True
        args.no_table_structure = True
    if args.no_ocr:
        args.ocr_mode = "off"

    if not args.input_dir.is_dir():
        print(f"ОШИБКА: {args.input_dir} не является директорией", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(args.input_dir.glob("*.pdf"))

    if args.only_docs:
        wanted: set[int] = set()
        for part in re.split(r"[,\s]+", args.only_docs.strip()):
            if not part:
                continue
            try:
                wanted.add(int(part))
            except ValueError:
                continue
        if wanted:
            pdf_files = [p for p in pdf_files if (_doc_num_from_stem(p.stem) in wanted)]

    if args.max_files is not None:
        pdf_files = pdf_files[: args.max_files]

    if not pdf_files:
        print(f"ОШИБКА: нет PDF-файлов в {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    if args.skip_existing:
        before = len(pdf_files)
        pdf_files = [p for p in pdf_files if not (args.output_dir / f"{p.stem}.md").is_file()]
        skipped = before - len(pdf_files)
        if skipped:
            print(f"Пропуск (--skip-existing): уже есть {skipped} .md, к обработке {len(pdf_files)} PDF\n")
        if not pdf_files:
            print("Нечего обрабатывать (все .md уже есть).", flush=True)
            sys.exit(0)

    ocr_languages = [x.strip() for x in args.ocr_languages.split(",") if x.strip()]
    if not ocr_languages:
        ocr_languages = ["ru", "en"]

    if args.ocr_mode == "auto":
        converter_with_ocr = _build_converter(
            do_ocr=True,
            no_table_structure=args.no_table_structure,
            full_quality=args.full_quality,
            ocr_languages=ocr_languages,
        )
        converter_no_ocr = _build_converter(
            do_ocr=False,
            no_table_structure=args.no_table_structure,
            full_quality=args.full_quality,
            ocr_languages=ocr_languages,
        )
    else:
        use_ocr = args.ocr_mode == "on"
        converter = _build_converter(
            do_ocr=use_ocr,
            no_table_structure=args.no_table_structure,
            full_quality=args.full_quality,
            ocr_languages=ocr_languages,
        )

    dev = os.environ.get("DOCLING_DEVICE", "auto")
    print("Инициализация Docling…")
    print(f"Устройство (DOCLING_DEVICE): {dev}", flush=True)
    print(f"OCR mode: {args.ocr_mode}; OCR languages: {','.join(ocr_languages)}", flush=True)
    if args.no_table_structure:
        print(
            "Предупреждение: без TableFormer таблицы в PDF часто не попадут в Markdown как сетка |...|.",
            flush=True,
        )
    print(
        "Загрузка весов layout/OCR/table (один раз, при первом обращении к pipeline) — "
        "может занять несколько минут; не прерывайте этот шаг (иначе придётся грузить снова).",
        flush=True,
    )
    if args.ocr_mode == "auto":
        converter_with_ocr.initialize_pipeline(InputFormat.PDF)
        converter_no_ocr.initialize_pipeline(InputFormat.PDF)
    else:
        converter.initialize_pipeline(InputFormat.PDF)
    print(f"Найдено {len(pdf_files)} PDF-файлов\n")

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {pdf_path.name}...", end=" ", flush=True)
        try:
            if args.ocr_mode == "auto":
                has_text_layer = _pdf_has_text_layer(pdf_path)
                active_converter = converter_no_ocr if has_text_layer else converter_with_ocr
                mode_name = "no-ocr(text-layer)" if has_text_layer else "ocr(scan)"
                print(f"{mode_name} ", end="", flush=True)
            else:
                active_converter = converter
            convert_pdf(
                pdf_path,
                args.output_dir,
                active_converter,
                text_cleanup=not args.no_text_cleanup,
            )
            # Fallback для auto: если текстовый слой «битый» и md вышел пустым — повторить с OCR.
            if args.ocr_mode == "auto" and has_text_layer:
                out_md = args.output_dir / f"{pdf_path.stem}.md"
                if _md_looks_empty(out_md):
                    print("fallback-ocr ", end="", flush=True)
                    convert_pdf(
                        pdf_path,
                        args.output_dir,
                        converter_with_ocr,
                        text_cleanup=not args.no_text_cleanup,
                    )
            print("OK")
        except Exception as e:
            print(f"ОШИБКА: {e}")
        finally:
            _clear_cuda_cache()

    print(f"\nГотово! Результаты: {args.output_dir}")


if __name__ == "__main__":
    main()
