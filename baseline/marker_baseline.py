from __future__ import annotations

import argparse
import gc
import os
import re
import sys
from pathlib import Path

os.environ.setdefault(
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "1",
)
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")


def _apply_device_from_argv() -> None:
    """``TORCH_DEVICE`` читается marker при импорте settings — задаём до импортов marker."""
    for i, arg in enumerate(sys.argv):
        if arg == "--device" and i + 1 < len(sys.argv):
            val = sys.argv[i + 1]
            if val != "auto":
                os.environ["TORCH_DEVICE"] = val
            return
        if arg.startswith("--device="):
            val = arg.split("=", 1)[1]
            if val != "auto":
                os.environ["TORCH_DEVICE"] = val
            return


_apply_device_from_argv()

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import convert_if_not_rgb, text_from_rendered
from marker.settings import settings

import torch


def _clear_cuda_cache() -> None:
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


def _rewrite_image_paths(markdown: str, old_to_new: dict[str, str]) -> str:
    """Подменить ссылки marker на ``images/doc_<n>_image_<k>.<ext>``."""
    out = markdown
    for old_name, new_basename in sorted(
        old_to_new.items(), key=lambda kv: len(kv[0]), reverse=True
    ):
        prefixed = f"images/{new_basename}"
        escaped = re.escape(old_name)
        out = re.sub(rf"\]\(\s*\./{escaped}\s*\)", f"]({prefixed})", out)
        out = re.sub(rf"\]\(\s*{escaped}\s*\)", f"]({prefixed})", out)
    return out


def _attention_implementation() -> str | None:
    """SDPA обычно быстрее eager на CUDA; на CPU оставляем дефолт модели."""
    if settings.TORCH_DEVICE_MODEL == "cuda":
        return "sdpa"
    return None


def _build_artifact_dict():
    device = settings.TORCH_DEVICE_MODEL
    dtype = settings.MODEL_DTYPE
    return create_model_dict(
        device=device,
        dtype=dtype,
        attention_implementation=_attention_implementation(),
    )


def convert_pdf(
    pdf_path: Path,
    output_dir: Path,
    artifact_dict: dict,
    disable_tqdm: bool = True,
    no_ocr: bool = False,
    fast_dpi: bool = True,
) -> None:
    stem = pdf_path.stem
    doc_num = _doc_num_from_stem(stem)
    if doc_num is None:
        doc_num = 1

    config: dict = {"disable_tqdm": disable_tqdm}
    if no_ocr:
        config["disable_ocr"] = True
    if fast_dpi:
        config["lowres_image_dpi"] = 88
        config["highres_image_dpi"] = 144

    converter = PdfConverter(
        config=config,
        artifact_dict=artifact_dict,
        processor_list=None,
        renderer=None,
        llm_service=None,
    )
    rendered = converter(str(pdf_path))
    text, _ext, images = text_from_rendered(rendered)
    text = text.encode(settings.OUTPUT_ENCODING, errors="replace").decode(
        settings.OUTPUT_ENCODING
    )

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    fmt = settings.OUTPUT_IMAGE_FORMAT or "PNG"
    ext = str(fmt).lower()
    if ext == "jpeg":
        ext = "jpg"

    old_to_new: dict[str, str] = {}
    order = 1
    for img_name, pil_img in images.items():
        new_basename = f"doc_{doc_num}_image_{order}.{ext}"
        old_to_new[img_name] = new_basename
        safe = images_dir / new_basename
        safe.parent.mkdir(parents=True, exist_ok=True)
        img = convert_if_not_rgb(pil_img)
        img.save(str(safe), fmt)
        order += 1

    text = _rewrite_image_paths(text, old_to_new)
    out_md = output_dir / f"{stem}.md"
    out_md.write_text(text, encoding=settings.OUTPUT_ENCODING)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline (marker-pdf): PDF → Markdown")
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
        help="Не запускать OCR (быстрее на PDF с текстовым слоем; сканы будут хуже)",
    )
    parser.add_argument(
        "--full-dpi",
        action="store_true",
        help="Стандартные DPI страниц (96/192), медленнее",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Устройство для моделей marker (auto = по умолчанию marker). "
        "При CUDA OOM используйте cpu (медленнее, без VRAM).",
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

    print("Загрузка моделей marker (один раз)…")
    print(
        f"Устройство: {settings.TORCH_DEVICE_MODEL}, dtype: {settings.MODEL_DTYPE}, "
        f"SDPA: {_attention_implementation() or 'по умолчанию'}",
        flush=True,
    )
    artifact_dict = _build_artifact_dict()
    print(f"Найдено {len(pdf_files)} PDF-файлов\n")

    fast_dpi = not args.full_dpi
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {pdf_path.name}...", end=" ", flush=True)
        try:
            convert_pdf(
                pdf_path,
                args.output_dir,
                artifact_dict,
                no_ocr=args.no_ocr,
                fast_dpi=fast_dpi,
            )
            print("OK")
        except Exception as e:
            print(f"ОШИБКА: {e}")
        finally:
            _clear_cuda_cache()

    print(f"\nГотово! Результаты: {args.output_dir}")


if __name__ == "__main__":
    main()
