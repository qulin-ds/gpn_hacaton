# PDF Parsing Baseline (Docling)

Baseline-решение для хакатона по парсингу PDF в Markdown на основе [Docling](https://github.com/docling-project/docling).

## Структура репозитория

```
├── dataset/
│   └── public/
│       ├── pdfs/              — 100 PDF-документов (document_001.pdf … document_100.pdf)
│       └── ground_truth/      — эталонные Markdown + images/
├── baseline/
│   └── docling_baseline.py    — baseline на Docling
├── pyproject.toml
└── README.md
```

## Быстрый старт

### 1. Установка зависимостей

Установите [uv](https://docs.astral.sh/uv/getting-started/installation/), затем:

```bash
uv sync
```

### 2. Запуск baseline

Обработать все 100 PDF и сохранить результаты в `output/`:

```bash
uv run baseline-docling \
    --input-dir dataset/public/pdfs \
    --output-dir dataset/public/baseline_results/
```

Для отладки можно ограничить число файлов:

```bash
uv run baseline-docling \
    --input-dir dataset/public/pdfs \
    --output-dir dataset/public/baseline_results/ \
    --max-files 5
```

Продолжить после обрыва (Ctrl+C): не трогать уже записанные `.md` и запустить с `--skip-existing`.

```bash
uv run baseline-docling \
    --input-dir dataset/public/pdfs \
    --output-dir dataset/public/baseline_results/ \
    --skip-existing
```

При первом запуске baseline **заранее** загружает веса layout/OCR/table; это может занять несколько минут — не прерывайте процесс на этом этапе, иначе при следующем запуске загрузка начнётся снова.


### Скорость и качество

По умолчанию baseline задаёт для Docling:

- `images_scale=0.88` и быстрый режим TableFormer (`FAST`);
- извлечение встроенных картинок (`generate_picture_images=True`).

Максимально быстро на PDF с нормальным текстовым слоем (без сканов и без тяжёлого TableFormer):

```bash
uv run baseline-docling \
    --input-dir dataset/public/pdfs \
    --output-dir dataset/public/baseline_results/ \
    --no-ocr \
    --no-table-structure
```

Если при старте падает импорт/инициализация таблиц (например, ошибка вокруг `cv2`), используйте `--no-table-structure`.

Полное качество (медленнее):

```bash
uv run baseline-docling ... --full-quality
```

Устройство для инференса (кроме `auto` задаётся до импорта Docling):

```bash
uv run baseline-docling ... --device cpu
```

## Формат решения

Решение принимает директорию с PDF и создаёт директорию с `.md`-файлами:

```
output/
├── document_001.md
├── document_002.md
├── ...
└── images/
    ├── doc_1_image_1.png
    ├── doc_1_image_2.png
    ├── doc_2_image_1.png
    └── ...
```

Имена `.md`-файлов должны совпадать с PDF (`document_001.pdf` → `document_001.md`).
Изображения — в подкаталоге `images/` с именами `doc_<n>_image_<k>.<ext>`, где `n` — номер документа без ведущих нулей, `k` — порядковый номер рисунка.
