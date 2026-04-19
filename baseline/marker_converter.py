#!/usr/bin/env python3
"""
Конвертер PDF → Markdown на базе Marker 1.10.2 + Ollama.
Использует правильную сигнатуру PdfConverter.
"""

from __future__ import annotations

import argparse
import base64
import gc
import os
import re
import sys
from pathlib import Path
from typing import Dict

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONWARNINGS"] = "ignore"


def _get_doc_id_from_stem(stem: str) -> str:
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        try:
            return str(int(parts[1]))
        except ValueError:
            pass
    return "0"


def _clear_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _fix_word_boundaries(text: str) -> str:
    html_entities = {
        "&amp;": "&", "&lt;": "<", "&gt;": ">", "&nbsp;": " ",
        "&quot;": '"', "&#39;": "'", "&apos;": "'", "&laquo;": '"',
        "&raquo;": '"', "&ldquo;": '"', "&rdquo;": '"',
        "&mdash;": "—", "&ndash;": "–",
    }
    for entity, replacement in html_entities.items():
        text = text.replace(entity, replacement)

    lines = text.split("\n")
    result_lines = []

    for line in lines:
        if line.strip().startswith("#"):
            line = re.sub(r"\s+", " ", line)
            result_lines.append(line)
            continue

        line = re.sub(r"([a-z])([A-Z])", r"\1 \2", line)
        line = re.sub(r"([а-яё])([А-ЯЁ])", r"\1 \2", line)
        line = re.sub(r"(\d)\s+([%°€$£¥])", r"\1\2", line)
        line = re.sub(r"(\d)\s+([.,])\s*(\d)", r"\1\2\3", line)
        line = re.sub(r"(\d{1,2})\s+([./-])\s*(\d{1,2})", r"\1\2\3", line)
        line = re.sub(r"(\d{1,2}[./-]\d{1,2})\s+([./-])\s*(\d{2,4})", r"\1\2\3", line)
        line = re.sub(
            r"(\w{2,})\s+([а-яёa-z]{1,3})(?=\s|$|[,.;:!?)])",
            r"\1\2", line, flags=re.IGNORECASE
        )
        line = re.sub(r"\s+", " ", line)
        line = re.sub(r"\s+([,.;:!?)])", r"\1", line)
        line = re.sub(r"([,.;:!?])([^\s\d])", r"\1 \2", line)

        result_lines.append(line)

    return "\n".join(result_lines)


def _clean_garbage_text(text: str) -> str:
    lines = text.split("\n")
    line_freq: Dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped) > 5:
            line_freq[stripped] = line_freq.get(stripped, 0) + 1

    garbage_patterns = [
        r"\bDRAFT\b", r"\bЧЕРНОВИК\b", r"\bCOPY\b", r"\bSAMPLE\b",
        r"\bCONFIDENTIAL\b", r"\bDO NOT COPY\b", r"\bPREVIEW\b",
        r"Patel-Stanley", r"Hooper and Sons", r"Morgan-Schwartz",
        r"Object-based intangible hub", r"Re-contextualized",
        r"Глубокий и третичный подход", r"^стр\.\s*\d+$", r"^Page\s*\d+$",
        r"^\d+\s*$", r"^\[\s*\d+\s*\]$", r"^\d+\s*/\s*\d+$",
    ]

    cleaned_lines = []
    consecutive_empty = 0
    max_empty = 2

    for line in lines:
        stripped = line.strip()

        if stripped in line_freq and line_freq[stripped] > 3:
            continue

        is_garbage = False
        for pattern in garbage_patterns:
            if re.search(pattern, stripped, re.IGNORECASE):
                if len(stripped) < 80:
                    is_garbage = True
                    break
                elif not any(m in stripped.lower() for m in ["глава", "раздел", "таблица", "рис."]):
                    is_garbage = True
                    break

        if is_garbage:
            continue

        if re.search(r"\d{2}[./-]\d{2}[./-]\d{4}", stripped):
            if re.search(r"стр\.|page|\[\s*\d+\s*\]|\d+\s*/\s*\d+", stripped.lower()):
                continue

        if not stripped:
            consecutive_empty += 1
            if consecutive_empty <= max_empty:
                cleaned_lines.append(line)
        else:
            consecutive_empty = 0
            cleaned_lines.append(line)

    while cleaned_lines and not cleaned_lines[0].strip():
        cleaned_lines.pop(0)
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()

    return "\n".join(cleaned_lines)


def convert_pdf_with_marker_ollama(pdf_path: Path, output_dir: Path, converter) -> None:
    stem = pdf_path.stem
    doc_id = _get_doc_id_from_stem(stem)

    images_out_dir = output_dir / "images"
    images_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Converting {stem}...", end=" ", flush=True)

    try:
        from marker.output import text_from_rendered

        rendered = converter(str(pdf_path))
        text, _, images = text_from_rendered(rendered)

        text = _fix_word_boundaries(text)
        text = _clean_garbage_text(text)

        image_map = {}
        if images:
            for img_name, img_data in images.items():
                img_bytes = base64.b64decode(img_data)
                counter = len(image_map) + 1
                new_filename = f"doc_{doc_id}_image_{counter}.png"
                dst_path = images_out_dir / new_filename
                dst_path.write_bytes(img_bytes)
                image_map[img_name] = new_filename

                old_pattern = f"![]({img_name})"
                new_pattern = f"![Image](images/{new_filename})"
                text = text.replace(old_pattern, new_pattern)

                old_pattern2 = f"![](images/{img_name})"
                text = text.replace(old_pattern2, new_pattern)

        out_md = output_dir / f"{stem}.md"
        out_md.write_text(text, encoding="utf-8")
        print("OK")

    except Exception as e:
        print(f"ERROR: {e}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="GPN Hackathon Marker + Ollama Converter")
    parser.add_argument("--input-dir", type=Path, required=True, help="Папка с PDF")
    parser.add_argument("--output-dir", type=Path, required=True, help="Папка результатов")
    parser.add_argument("--max-files", type=int, default=None, help="Лимит файлов")
    parser.add_argument("--device", type=str, default="cpu", help="cuda / cpu / auto")
    parser.add_argument("--force-ocr", action="store_true", help="Принудительный OCR")
    parser.add_argument("--use-llm", action="store_true", help="Использовать локальную LLM через Ollama")
    parser.add_argument("--ollama-model", type=str, default="llava:13b", help="Модель Ollama")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="URL сервера Ollama")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "images").mkdir(exist_ok=True)

    pdf_files = sorted(args.input_dir.glob("*.pdf"))
    if args.max_files:
        pdf_files = pdf_files[:args.max_files]

    if not pdf_files:
        print("Нет PDF файлов.")
        return

    print(f"Найдено {len(pdf_files)} файлов. Инициализация моделей Marker...")

    # Определение устройства
    device = args.device
    if device in ("cuda", "auto"):
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                print("CUDA доступна, используется GPU")
            else:
                device = "cpu"
                print("CUDA недоступна, переключаемся на CPU")
        except ImportError:
            device = "cpu"
            print("PyTorch не найден, используется CPU")
    else:
        device = "cpu"
        print("Используется CPU")

    from marker.config.parser import ConfigParser
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    config = {
        "output_format": "markdown",
        "force_ocr": args.force_ocr,
        "use_llm": args.use_llm,
        "extract_images": True,
        "device": device,
        "languages": ["ru", "en"],
    }

    if args.use_llm:
        config["llm_service"] = "marker.services.ollama.OllamaService"
        config["ollama_model"] = args.ollama_model
        config["ollama_base_url"] = args.ollama_url
        print(f"⚙️  Используется локальный LLM: {args.ollama_model} через {args.ollama_url}")

    config_parser = ConfigParser(config)
    config_dict = config_parser.generate_config_dict()
    renderer = config_parser.get_renderer()

    # Создаём artifact_dict через create_model_dict (единственный доступный способ)
    artifact_dict = create_model_dict(device=device)

    # Создаём конвертер с правильной сигнатурой: artifact_dict, затем config=..., renderer=...
    converter = PdfConverter(
        artifact_dict,
        config=config_dict,
        renderer=renderer,
        llm_service=config.get("llm_service") if args.use_llm else None,
    )

    print("Модели загружены. Начинаем обработку...")

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {pdf_path.name}", end="", flush=True)
        try:
            convert_pdf_with_marker_ollama(pdf_path, args.output_dir, converter)
        except Exception as e:
            print(f"ERROR: {e}")
        finally:
            _clear_cuda_cache()

    print(f"\nГотово! Результаты в: {args.output_dir}")


if __name__ == "__main__":
    main()