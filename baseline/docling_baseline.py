from __future__ import annotations

import argparse
import gc
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")


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

_IMG_LINK_RE = re.compile(
    r"images/(image_\d+_[a-f0-9]+\.(?:png|jpe?g))",
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
    """Как в evaluation: ``document_051`` → 51."""
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _move_or_convert_to_png(src: Path, dst: Path) -> None:
    """Сохранить только PNG: JPEG конвертируем, остальное переносим как есть."""
    ext = src.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        from PIL import Image

        with Image.open(src) as im:
            im.save(dst, format="PNG")
        src.unlink()
    else:
        shutil.move(str(src), str(dst))


def _normalize_image_names(
    markdown: str,
    work_images_dir: Path,
    out_images_dir: Path,
    doc_num: int,
) -> str:
    """Переименовать ``image_*_*.(png|jpg)`` в ``doc_<n>_image_<k>.png`` и обновить ссылки."""
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
        _move_or_convert_to_png(src, out_images_dir / new_name)
        order += 1

    out = markdown
    for old_name, new_name in sorted(old_to_new.items(), key=lambda kv: len(kv[0]), reverse=True):
        out = out.replace(f"images/{old_name}", f"images/{new_name}")
    return out


def _build_converter(
    no_ocr: bool,
    no_table_structure: bool,
    full_quality: bool,
) -> DocumentConverter:
    if full_quality:
        images_scale = 1.0
        table_opts = TableStructureOptions(mode=TableFormerMode.ACCURATE)
    else:
        images_scale = 0.88
        table_opts = TableStructureOptions(mode=TableFormerMode.FAST)

    pipeline_options = ThreadedPdfPipelineOptions(
        do_ocr=not no_ocr,
        do_table_structure=not no_table_structure,
        generate_picture_images=True,
        images_scale=images_scale,
        table_structure_options=table_opts,
        accelerator_options=AcceleratorOptions(),
    ).model_copy(deep=True)
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        },
    )


def convert_pdf(
    pdf_path: Path,
    output_dir: Path,
    converter: DocumentConverter,
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
        doc.save_as_markdown(
            md_work,
            artifacts_dir=Path("images"),
            image_mode=ImageRefMode.REFERENCED,
        )
        text = md_work.read_text(encoding="utf-8")
        text = _normalize_image_names(
            text,
            work_images_dir=work / "images",
            out_images_dir=output_dir / "images",
            doc_num=doc_num,
        )
        out_md = output_dir / f"{stem}.md"
        out_md.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline (Docling): PDF → Markdown")
    parser.add_argument("--input-dir", type=Path, required=True, help="Директория с PDF")
    parser.add_argument("--output-dir", type=Path, required=True, help="Каталог результатов")
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
        "--no-table-structure",
        action="store_true",
        help="Отключить TableFormer (быстрее; при проблемах с opencv/cv2)",
    )
    parser.add_argument(
        "--full-quality",
        action="store_true",
        help="Полное качество: images_scale=1.0 и точные таблицы (медленнее)",
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
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"ОШИБКА: {args.input_dir} не является директорией", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(args.input_dir.glob("*.pdf"))
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

    dev = os.environ.get("DOCLING_DEVICE", "auto")
    print("Инициализация Docling…")
    print(f"Устройство (DOCLING_DEVICE): {dev}", flush=True)
    converter = _build_converter(
        no_ocr=args.no_ocr,
        no_table_structure=args.no_table_structure,
        full_quality=args.full_quality,
    )
    print(
        "Загрузка весов layout/OCR/table (один раз, при первом обращении к pipeline) — "
        "может занять несколько минут; не прерывайте этот шаг (иначе придётся грузить снова).",
        flush=True,
    )
    converter.initialize_pipeline(InputFormat.PDF)
    print(f"Найдено {len(pdf_files)} PDF-файлов\n")

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {pdf_path.name}...", end=" ", flush=True)
        try:
            convert_pdf(pdf_path, args.output_dir, converter)
            print("OK")
        except Exception as e:
            print(f"ОШИБКА: {e}")
        finally:
            _clear_cuda_cache()

    print(f"\nГотово! Результаты: {args.output_dir}")


if __name__ == "__main__":
    main()
