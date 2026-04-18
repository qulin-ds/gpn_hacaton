from __future__ import annotations

import argparse
import gc
import hashlib
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List

# Подавляем шумные логи библиотек
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

def _apply_device_from_argv() -> None:
    """Передача параметра устройства в переменную окружения для Docling."""
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
    """Хак для совместимости некоторых версий cv2."""
    try:
        import cv2
        if not hasattr(cv2, "setNumThreads"):
            cv2.setNumThreads = lambda _nthreads: None
    except ImportError:
        pass

_patch_cv2_set_num_threads()

from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.base import ImageRefMode

def _clear_cuda_cache() -> None:
    """Очистка памяти GPU после обработки файла."""
    try:
        import torch
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
    except ImportError:
        pass

def _get_doc_id_from_stem(stem: str) -> str:
    """
    Извлекает ID документа из имени файла document_NNN.
    Возвращает число без ведущих нулей (например, 'document_051' -> '51').
    """
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        try:
            return str(int(parts[1]))
        except ValueError:
            pass
    return "0"

def _fix_merged_cells_in_markdown(text: str) -> str:
    """
    Исправляет объединенные ячейки в Markdown таблицах.
    Разбивает объединенные ячейки на индивидуальные с копированием содержимого.
    """
    lines = text.split('\n')
    result_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Проверяем, является ли строка частью таблицы
        if line.strip().startswith('|') and line.strip().endswith('|'):
            # Получаем контекст таблицы (несколько строк)
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i])
                i += 1
            
            # Обрабатываем таблицу
            processed_table = _process_table_merged_cells(table_lines)
            result_lines.extend(processed_table)
        else:
            result_lines.append(line)
            i += 1
    
    return '\n'.join(result_lines)

def _process_table_merged_cells(table_lines: List[str]) -> List[str]:
    """Обрабатывает одну таблицу, исправляя объединенные ячейки."""
    if not table_lines:
        return table_lines
    
    # Определяем количество столбцов по строке-разделителю или по максимальному количеству ячеек
    max_cols = 0
    
    for line in table_lines:
        if re.match(r'^[\s\|:\-]+$', line.strip()):
            cols = len([c for c in line.split('|') if c.strip() or c == ''])
            max_cols = max(max_cols, cols)
        else:
            parts = line.split('|')
            cols = len([p for p in parts if p or p == ''])
            max_cols = max(max_cols, cols)
    
    if max_cols == 0:
        return table_lines
    
    processed = []
    
    for line in table_lines:
        if re.match(r'^[\s\|:\-]+$', line.strip()):
            processed.append(line)
            continue
        
        parts = line.split('|')
        if parts and parts[0] == '':
            parts.pop(0)
        if parts and parts[-1] == '':
            parts.pop(-1)
        
        # Если ячеек меньше, чем должно быть - дублируем последнюю
        while len(parts) < max_cols:
            if parts:
                parts.append(parts[-1])
            else:
                parts.append('')
        
        # Если ячеек больше - обрезаем
        if len(parts) > max_cols:
            parts = parts[:max_cols]
        
        processed.append('|' + '|'.join(parts) + '|')
    
    return processed

def _fix_multilevel_headers(text: str) -> str:
    """
    Исправляет многоуровневые заголовки таблиц.
    Объединяет через нижнее подчеркивание.
    """
    lines = text.split('\n')
    result_lines = []
    i = 0
    
    while i < len(lines):
        if lines[i].strip().startswith('|'):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i])
                i += 1
            
            if len(table_lines) >= 2:
                processed = _merge_multilevel_headers(table_lines)
                result_lines.extend(processed)
            else:
                result_lines.extend(table_lines)
        else:
            result_lines.append(lines[i])
            i += 1
    
    return '\n'.join(result_lines)

def _merge_multilevel_headers(table_lines: List[str]) -> List[str]:
    """Объединяет многоуровневые заголовки."""
    if len(table_lines) < 3:
        return table_lines
    
    separator_idx = -1
    for idx, line in enumerate(table_lines):
        if re.match(r'^[\s\|:\-]+$', line.strip()):
            separator_idx = idx
            break
    
    if separator_idx <= 1:
        return table_lines
    
    header_lines = table_lines[:separator_idx]
    if len(header_lines) >= 2:
        headers = []
        for line in header_lines:
            parts = [p.strip() for p in line.split('|') if p.strip() or p == '']
            if parts and parts[0] == '':
                parts.pop(0)
            if parts and parts[-1] == '':
                parts.pop(-1)
            headers.append(parts)
        
        max_cols = max(len(h) for h in headers)
        for h in headers:
            while len(h) < max_cols:
                h.append('')
        
        merged_parts = []
        for col_idx in range(max_cols):
            col_headers = []
            for row_idx in range(len(headers)):
                if col_idx < len(headers[row_idx]) and headers[row_idx][col_idx].strip():
                    col_headers.append(headers[row_idx][col_idx].strip())
            merged_parts.append('_'.join(col_headers) if col_headers else '')
        
        new_header_line = '| ' + ' | '.join(merged_parts) + ' |'
        result = [new_header_line]
        result.extend(table_lines[separator_idx:])
        return result
    
    return table_lines

def _fix_broken_words(text: str) -> str:
    """Исправляет артефакты PDF-парсинга."""
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")
    
    lines = text.split('\n')
    fixed_lines = []
    
    for line in lines:
        if line.strip().startswith('#'):
            line = re.sub(r'(\d)\s+(\d)', r'\1\2', line)
            fixed_lines.append(line)
            continue
        
        line = re.sub(r'([a-zа-яё])([A-ZА-ЯЁ])', r'\1 \2', line)
        line = re.sub(r'(\d)\s+(\d)', r'\1\2', line)
        line = re.sub(r'(\d)\s+([.,])\s*(\d)', r'\1\2\3', line)
        line = re.sub(r'(\w)\s+([a-zа-яё]{1,3})(?=\W|$)', r'\1\2', line)
        line = re.sub(r'\s+([.,;:!?])', r'\1', line)
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def _remove_headers_footers_and_watermarks(text: str) -> str:
    """Удаляет колонтитулы, водяные знаки и повторяющийся мусор."""
    lines = text.split('\n')
    cleaned_lines = []
    
    garbage_patterns = [
        r"Patel-Stanley",
        r"Hooper and Sons",
        r"Morgan-Schwartz",
        r"Object-based intangible hub",
        r"Re-contextualized zero tolerance analyzer",
        r"Глубокий и третичный подход",
        r"ЧЕРНОВИК",
        r"\bDRAFT\b",
        r"стр\.\s*\d+",
        r"Page\s*\d+",
        r"\[\s*\d+\s*\]",
        r"\d+\s*/\s*\d+",
        r"\.{3,}",
    ]
    
    line_counts = {}
    for line in lines:
        stripped = line.strip()
        if stripped:
            line_counts[stripped] = line_counts.get(stripped, 0) + 1
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue
        
        if line_counts.get(stripped, 0) > 3:
            continue
        
        is_garbage = False
        for pattern in garbage_patterns:
            if re.search(pattern, stripped, re.IGNORECASE):
                if len(stripped) < 100 and not stripped.startswith('#'):
                    is_garbage = True
                    break
        
        if is_garbage:
            continue
        
        if re.search(r'\d{2}\.\d{2}\.\d{4}', stripped):
            if re.search(r'(стр\.|Page|\[\s*\d+\s*\]|\d+\s*/\s*\d+)', stripped):
                continue
        
        if '·' in stripped and len(stripped) < 100:
            continue
        
        cleaned_lines.append(line)
    
    while cleaned_lines and not cleaned_lines[0].strip():
        cleaned_lines.pop(0)
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()
    
    return '\n'.join(cleaned_lines)

def _fix_table_formatting(text: str) -> str:
    """Исправляет форматирование таблиц."""
    lines = text.split('\n')
    result_lines = []
    i = 0
    
    while i < len(lines):
        if lines[i].strip().startswith('|'):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i])
                i += 1
            
            processed = _normalize_table(table_lines)
            result_lines.extend(processed)
        else:
            result_lines.append(lines[i])
            i += 1
    
    return '\n'.join(result_lines)

def _normalize_table(table_lines: List[str]) -> List[str]:
    """Нормализует таблицу - выравнивает столбцы и добавляет разделители."""
    if not table_lines:
        return table_lines
    
    max_cols = 0
    has_separator = False
    
    for line in table_lines:
        if re.match(r'^[\s\|:\-]+$', line.strip()):
            has_separator = True
        parts = line.split('|')
        if parts and parts[0] == '':
            parts.pop(0)
        if parts and parts[-1] == '':
            parts.pop(-1)
        max_cols = max(max_cols, len(parts))
    
    if max_cols == 0:
        return table_lines
    
    result = []
    separator_added = False
    
    for line in table_lines:
        if re.match(r'^[\s\|:\-]+$', line.strip()):
            result.append(line)
            separator_added = True
            continue
        
        parts = line.split('|')
        if parts and parts[0] == '':
            parts.pop(0)
        if parts and parts[-1] == '':
            parts.pop(-1)
        
        while len(parts) < max_cols:
            parts.append('')
        
        if len(parts) > max_cols:
            parts = parts[:max_cols]
        
        result.append('| ' + ' | '.join(p.strip() for p in parts) + ' |')
    
    if not separator_added and len(result) > 0:
        separator = '|' + '|'.join([' --- ' for _ in range(max_cols)]) + '|'
        result.insert(1, separator)
    
    return result

def _extract_text_from_raster_regions(text: str) -> str:
    """Улучшает извлечение текста из растровых областей."""
    ocr_fixes = {
        'º': '°', 'ª': '°', '±': '±', '×': '×', '÷': '/',
        '≈': '≈', '≤': '≤', '≥': '≥', '←': '←', '→': '→',
        '·': '·', '«': '"', '»': '"', '„': '"', '"': '"',
        ''': "'", ''': "'", '—': '-', '–': '-', '…': '...',
    }
    
    for wrong, correct in ocr_fixes.items():
        text = text.replace(wrong, correct)
    
    text = re.sub(r'([а-яё])\s+([а-яё])', r'\1\2', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    
    return text

def convert_pdf(pdf_path: Path, output_dir: Path, converter: DocumentConverter) -> None:
    """Конвертирует PDF в Markdown с полной пост-обработкой."""
    stem = pdf_path.stem
    doc_id = _get_doc_id_from_stem(stem)
    
    images_out_dir = output_dir / "images"
    images_out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        result = converter.convert(str(pdf_path))
        doc = result.document
        
        with tempfile.TemporaryDirectory(prefix=f"docling_{stem}_") as tmp:
            work = Path(tmp)
            md_work = work / f"{stem}.md"
            
            doc.save_as_markdown(
                md_work,
                artifacts_dir=work / "images",
                image_mode=ImageRefMode.REFERENCED,
            )
            
            text = md_work.read_text(encoding="utf-8")
            
            # Пост-обработка
            text = _extract_text_from_raster_regions(text)
            text = _fix_broken_words(text)
            text = _remove_headers_footers_and_watermarks(text)
            text = _fix_multilevel_headers(text)
            text = _fix_merged_cells_in_markdown(text)
            text = _fix_table_formatting(text)
            
            # Обработка изображений
            artifact_dir = work / "images"
            if artifact_dir.exists():
                img_files = sorted(artifact_dir.glob("*"))
                counter = 1
                image_map = {}
                
                for old_img_path in img_files:
                    if not old_img_path.is_file():
                        continue
                    
                    ext = old_img_path.suffix.lower()
                    if ext not in ['.png', '.jpg', '.jpeg']:
                        ext = '.png'
                    
                    new_filename = f"doc_{doc_id}_image_{counter}{ext}"
                    dst_path = images_out_dir / new_filename
                    
                    if dst_path.exists():
                        with open(old_img_path, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()[:8]
                        new_filename = f"doc_{doc_id}_image_{counter}_{file_hash}{ext}"
                        dst_path = images_out_dir / new_filename
                    
                    shutil.copy2(old_img_path, dst_path)
                    image_map[old_img_path.name] = new_filename
                    counter += 1
                
                for old_name, new_name in image_map.items():
                    new_rel_path = f"images/{new_name}"
                    
                    patterns = [
                        rf'!\[.*?\]\(.*?{re.escape(old_name)}\)',
                        rf'!\[.*?\]\(\.\/images\/{re.escape(old_name)}\)',
                        rf'!\[.*?\]\(images\/{re.escape(old_name)}\)',
                        rf'!\[.*?\]\({re.escape(old_name)}\)',
                    ]
                    
                    for pattern in patterns:
                        def replace_match(match, alt_text=None):
                            if alt_text is None:
                                alt_match = re.search(r'!\[(.*?)\]', match.group(0))
                                alt_text = alt_match.group(1) if alt_match else 'Image'
                            return f'![{alt_text}]({new_rel_path})'
                        
                        text = re.sub(pattern, lambda m: replace_match(m), text)
                    
                    if old_name in text:
                        text = text.replace(old_name, new_rel_path)
            
            out_md = output_dir / f"{stem}.md"
            out_md.write_text(text, encoding="utf-8")
            
    except Exception as e:
        print(f"ERROR in {stem}: {e}")
        raise

def main() -> None:
    parser = argparse.ArgumentParser(description="GPN Hackathon Improved Baseline")
    parser.add_argument("--input-dir", type=Path, required=True, help="Папка с PDF")
    parser.add_argument("--output-dir", type=Path, required=True, help="Папка результатов")
    parser.add_argument("--no-ocr", action="store_true", help="Отключить OCR")
    parser.add_argument("--no-table-structure", action="store_true", help="Отключить анализ таблиц")
    parser.add_argument("--max-files", type=int, default=None, help="Лимит файлов")
    parser.add_argument("--device", type=str, default="cpu", help="Устройство (cuda/cpu/auto)")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "images").mkdir(exist_ok=True)
    
    pdf_files = sorted(args.input_dir.glob("*.pdf"))
    if args.max_files:
        pdf_files = pdf_files[:args.max_files]
    
    if not pdf_files:
        print("Нет PDF файлов.")
        return
    
    print(f"Найдено {len(pdf_files)} файлов. Инициализация модели...")
    
    # Определяем устройство
    device_str = args.device
    if device_str == "auto":
        try:
            import torch
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device_str = "cpu"
    
    # Конвертируем строку в enum AcceleratorDevice
    device_map = {
        "cpu": AcceleratorDevice.CPU,
        "cuda": AcceleratorDevice.CUDA,
        "mps": AcceleratorDevice.MPS,
        "xpu": AcceleratorDevice.XPU,
    }
    accelerator_device = device_map.get(device_str, AcceleratorDevice.CPU)
    
    # Настройка пайплайна
    pipeline_options = PdfPipelineOptions(
        do_ocr=not args.no_ocr,
        do_table_structure=not args.no_table_structure,
        generate_picture_images=True,
        generate_page_images=False,
        images_scale=2.0,
        table_structure_options=TableStructureOptions(
            mode=TableFormerMode.ACCURATE,
            do_cell_matching=True,
        ),
        accelerator_options=AcceleratorOptions(
            num_threads=4,
            device=accelerator_device,
        ),
    )
    
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        },
    )
    
    # Обработка файлов
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {pdf_path.name}...", end=" ", flush=True)
        try:
            convert_pdf(pdf_path, args.output_dir, converter)
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
        finally:
            _clear_cuda_cache()
    
    print(f"\nГотово! Результаты в: {args.output_dir}")

if __name__ == "__main__":
    main()