from __future__ import annotations

import argparse
import gc
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

# Подавляем шумные логи библиотек
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

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
    except ImportError:
        return
    if not hasattr(cv2, "setNumThreads"):
        cv2.setNumThreads = lambda _nthreads: None

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

def _fix_broken_words(text: str) -> str:
    """
    Исправляет артефакты PDF-парсинга:
    1. Разделяет слипшиеся слова (OftenMy -> Often My).
    2. Склеивает разорванные даты/числа (03.03.1 999 -> 03.03.1999).
    3. Склеивает короткие окончания в таблицах (Наткнут ься -> Наткнуться).
    """
    lines = text.split('\n')
    fixed_lines = []
    
    for line in lines:
        # ЗАГОЛОВКИ MARKDOWN НЕ ТРОГАЕМ (чтобы не создать кашу из # ##)
        if line.strip().startswith('#'):
            # Только цифры в заголовках склеиваем
            line = re.sub(r'(\d)\s+(\d)', r'\1\2', line)
            fixed_lines.append(line)
            continue
            
        # --- ШАГ 1: Разделение слипшихся слов (CamelCase ошибки парсера) ---
        # Если строчная буква идет сразу перед Заглавной (латиница или кириллица), вставляем пробел
        # Пример: OftenMy -> Often My, SecurityArm -> Security Arm
        line = re.sub(r'([a-zа-яё])([A-ZА-ЯЁ])', r'\1 \2', line)
        
        # --- ШАГ 2: Склейка цифр (даты, деньги, ID) ---
        # Пример: 03.03.1 999 -> 03.03.1999, 897 659 -> 897659
        line = re.sub(r'(\d)\s+(\d)', r'\1\2', line)
        
        # --- ШАГ 3: Склейка коротких "хвостов" слов в таблицах ---
        # Логика: Если слово заканчивается на букву, затем пробел, 
        # затем короткий хвост (1-3 строчные буквы), и после хвоста конец строки или знак препинания.
        # Это чинит "Наткнут ься", "Переосмысленн ая", но не трогает "Дом Кот".
        line = re.sub(r'(\w)\s+([a-zа-яё]{1,3})(?=\W|$)', r'\1\2', line)
        
        fixed_lines.append(line)
        
    return '\n'.join(fixed_lines)

def _remove_headers_footers(text: str) -> str:
    """
    Удаляет колонтитулы и водяные знаки ГПН.
    Основано на паттернах из document_001.pdf и document_093.pdf.
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    # Паттерны мусора из ТЗ и примеров
    garbage_patterns = [
        r"Patel-Stanley",
        r"Hooper and Sons",
        r"Object-based intangible hub",
        r"Re-contextualized zero tolerance analyzer",
        r"ЧЕРНОВИК",
    ]
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue
            
        is_garbage = False
        
        # 1. Проверка по явным паттернам
        for pattern in garbage_patterns:
            if re.search(pattern, stripped, re.IGNORECASE):
                is_garbage = True
                break
                
        if is_garbage:
            continue

        # 2. Проверка на колонтитулы с датами и номерами страниц
        # Примеры: "Patel-Stanley · решение · 1998-06-25 стр. 1"
        #          "Hooper and Sons · привлекать · 2001-02-03 — 1 —"
        #          "Hooper and Sons · привлекать · 2001-02-03 [ 2 ]"
        #          "Hooper and Sons · привлекать · 2001-02-03 5/?"
        if re.search(r'\d{2}\.\d{2}\.\d{4}', stripped):
            if re.search(r'(стр\.|Page|— \d+ —|\[ \d+ \]|\d/\?)', stripped):
                is_garbage = True
        
        # 3. Проверка на короткие строки с разделителями '·' (типичный колонтитул)
        if '·' in stripped and len(stripped) < 150:
             is_garbage = True

        if not is_garbage:
            cleaned_lines.append(line)
            
    return '\n'.join(cleaned_lines)

def convert_pdf(pdf_path: Path, output_dir: Path, converter: DocumentConverter) -> None:
    stem = pdf_path.stem
    doc_id = _get_doc_id_from_stem(stem)
    
    # Папка для картинок всех документов
    images_out_dir = output_dir / "images"
    images_out_dir.mkdir(parents=True, exist_ok=True)

    result = converter.convert(str(pdf_path))
    doc = result.document

    with tempfile.TemporaryDirectory(prefix=f"docling_{stem}_") as tmp:
        work = Path(tmp)
        md_work = work / f"{stem}.md"
        
        # 1. Сохраняем Markdown во временную папку. 
        # Картинки будут сохранены в work/images/
        doc.save_as_markdown(
            md_work,
            artifacts_dir=work / "images", 
            image_mode=ImageRefMode.REFERENCED,
        )
        
        text = md_work.read_text(encoding="utf-8")
        
        # 2. Пост-обработка текста
        text = _fix_broken_words(text)
        text = _remove_headers_footers(text)
        
        # 3. Перенос картинок и коррекция ссылок
        artifact_dir = work / "images"
        if artifact_dir.exists():
            img_files = sorted(artifact_dir.glob("*"))
            counter = 1
            
            for old_img_path in img_files:
                if not old_img_path.is_file():
                    continue
                    
                ext = old_img_path.suffix.lower()
                if ext not in ['.png', '.jpg', '.jpeg']:
                    ext = '.png'
                    
                # Формат имени: doc_<ID>_image_<N>.png (ТЗ требует ID без ведущих нулей)
                new_filename = f"doc_{doc_id}_image_{counter}{ext}"
                dst_path = images_out_dir / new_filename
                
                # Копируем файл
                shutil.copy2(old_img_path, dst_path)
                
                # --- НАДЕЖНАЯ ЗАМЕНА ПУТЕЙ ---
                # Docling может вставить: image_1.png, ./image_1.png, C:/Temp/.../image_1.png
                # Мы должны заменить любое вхождение этого файла на images/new_filename
                
                old_name = old_img_path.name
                new_rel_path = f"images/{new_filename}"
                
                # Экранируем имя файла для использования в regex
                escaped_old_name = re.escape(old_name)
                
                # Паттерн ищет: необязательный путь (с прямыми или обратными слэшами), заканчивающийся именем файла
                # Заменяем на новый относительный путь
                # Также обрабатываем случай, когда ссылка уже частично заменена или имеет префикс
                pattern = r'(?:.*[\\/])?' + escaped_old_name
                text = re.sub(pattern, new_rel_path, text)
                
                # Дополнительная проверка: если в тексте осталось что-то вроде "images/image_1.png)" или "![](image_1.png)"
                # Регулярка выше должна это покрыть, но на всякий случай проверим явные вхождения имени файла
                if old_name in text:
                     text = text.replace(old_name, new_rel_path)

                counter += 1

        # 4. Сохраняем итоговый MD
        out_md = output_dir / f"{stem}.md"
        out_md.write_text(text, encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser(description="GPN Hackathon Baseline (Docling)")
    parser.add_argument("--input-dir", type=Path, required=True, help="Папка с PDF")
    parser.add_argument("--output-dir", type=Path, required=True, help="Папка результатов")
    parser.add_argument("--no-ocr", action="store_true", help="Отключить OCR (быстрее, но хуже текст)")
    parser.add_argument("--no-table-structure", action="store_true", help="Отключить анализ структуры таблиц")
    parser.add_argument("--max-files", type=int, default=None, help="Лимит файлов для теста")
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
    
    # Настройка пайплайна
    # Приоритет ТЗ: Таблицы > Текст. Поэтому используем ACCURATE для таблиц и включаем OCR.
    pipeline_options = ThreadedPdfPipelineOptions(
        do_ocr=not args.no_ocr,
        do_table_structure=not args.no_table_structure,
        generate_picture_images=True,
        images_scale=1.0, # Максимальное качество картинок
        table_structure_options=TableStructureOptions(
            mode=TableFormerMode.ACCURATE # Лучшее качество для сложных таблиц
        ),
        accelerator_options=AcceleratorOptions(),
    )
    
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        },
    )

    # Прогрев модели на первом файле
    if pdf_files:
        print("Прогрев модели (может занять 1-2 минуты)...")
        try:
            convert_pdf(pdf_files[0], args.output_dir, converter)
            print("Прогрев завершен. Обработка остальных файлов...\n")
            remaining_files = pdf_files[1:]
        except Exception as e:
            print(f"Ошибка при прогреве: {e}")
            remaining_files = pdf_files
    else:
        remaining_files = []

    for i, pdf_path in enumerate(remaining_files, 1):
        print(f"[{i}/{len(remaining_files)}] {pdf_path.name}...", end=" ", flush=True)
        try:
            convert_pdf(pdf_path, args.output_dir, converter)
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
        finally:
            _clear_cuda_cache()

    print("\nГотово! Результаты в:", args.output_dir)

if __name__ == "__main__":
    main()