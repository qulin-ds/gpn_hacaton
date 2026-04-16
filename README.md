# PDF Parsing Baseline (marker-pdf)

Baseline-решение для хакатона по парсингу PDF в Markdown на основе [marker-pdf](https://github.com/datalab-to/marker).

## Структура репозитория

```
├── dataset/
│   └── public/
│       ├── pdfs/              — 100 PDF-документов (document_001.pdf … document_100.pdf)
│       └── ground_truth/      — эталонные Markdown + images/
├── baseline/
│   └── marker_baseline.py     — baseline на marker-pdf
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
uv run baseline-marker \
    --input-dir dataset/public/pdfs \
    --output-dir dataset/public/baseline_results/
```

Для отладки можно ограничить число файлов:

```bash
uv run baseline-marker \
    --input-dir dataset/public/pdfs \
    --output-dir dataset/public/baseline_results/ \
    --max-files 5
```

### Скорость

По умолчанию baseline задаёт для marker:

- явные `device` и `dtype` из настроек marker (на CUDA — `bfloat16`);
- на CUDA — внимание через SDPA (`attention_implementation="sdpa"`);
- чуть пониженные DPI страниц (`88` / `144` вместо `96` / `192`), быстрее layout и детекция.

Максимально быстро на PDF с нормальным текстовым слоем (без сканов):

```bash
uv run baseline-marker \
    --input-dir dataset/public/pdfs \
    --output-dir dataset/public/baseline_results/ \
    --no-ocr
```

Вернуть стандартные DPI (медленнее, потенциально точнее):

```bash
uv run baseline-marker ... --full-dpi
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
