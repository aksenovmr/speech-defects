# Детекция дефектов речи с помощью скороговорок (WavLM-base encoder + MLP)

Сервис для детекции дефектов детской речи с помощью скороговорок. Модель определяет наличие дефектов речи и локализует их на уровне конкретных звуков.

Проверяемые звуки: **л, р, т, с, ш, ч, ц, щ**

## Структура проекта
```
speech-defects/
├── api/                    # FastAPI-сервер
├── configs/                # YAML-конфиги инференса и обучения
├── demo/                   # Streamlit веб-интерфейс
├── notebooks/              # Jupyter-ноутбук с baseline-решениями и экспериментами
├── scripts/                # Скрипт обучения
├── src/
│   ├── data/               # Датасет, даталоадеры
│   ├── inference/          # Пайплайн инференса
│   ├── models/             # Архитектура модели и лосс
│   ├── service/            # Хранилище результатов (SQLite)
│   ├── training/           # Trainer, метрики, логирование Wandb
│   └── utils/              # Загрузка конфигов
├── tests/                  # Тесты API
├── Dockerfile.api
├── Dockerfile.streamlit
└── docker-compose.yml
```

## Модель

Основная модель построена на базе предобученного речевого энкдера WavLM [microsoft/wavlm-base](https://huggingface.co/microsoft/wavlm-base). Задача решается в рамках парадигмы Multiple Instance Learning (MLP) с использованием weak supervision подхода. Модель обучается на бинарных метках качества записи (хорошо/плохо) с дополнительным использованием информации о звуках, проверяемых в скороговорках, без использования пофонемной разметки.

Чекпоинт: [aksenovmr/wavlm_base_unfreeze4_speech_defects](https://huggingface.co/aksenovmr/wavlm_base_unfreeze4_speech_defects)

Обучение основной модели, а также бейзлайн-модели и прочих экспериментов, логировалось в Wandb. Составленныый отчет доступен по ссылке: [wandb_report](https://api.wandb.ai/links/aksenovmr-hse-university/4uv8lnzh)

## Архитектура сервиса

Сервис состоит из двух компонентов, которые запускаются в отдельных Docker-контейнерах и взаимодействуют через HTTP.

**API-сервер (`api/`)** - бекэнд на FastAPI. Загружает модель из чекпоинта, принимает на вход аудиофайл, конвертирует его, запускает инференс, интерпретирует рехультат и сохраняет историю запросов в SQLite.

**Streamlit-интерфейс (`demo/`)** - веб-приложение для пользователей. Обращается к API через HTTP и визуализирует результаты.

**Хранилище (`src/service/storage.py`)**  - сохранение результатов запросов в базу данных SQLite. Сохраняется информация о вероятности дефекта, вердикте модели, проблемных звуках, оценка по каждому проверяемому звуку, имя и версия используемой для анализа модели, а также пороговые значения и идентификатор сессии. История доступна через эндпоинт `/history`.

## Как запустить сервис с помощью Docker?

### 1. Клонировать репозиторий

```bash
git clone https://github.com/aksenovmr/speech-defects.git
cd speech-defects
```

### 2. Запустить сервис

```bash
docker-compose up --build
```

При первом запуске чекпоинт модели автоматически скачается с Hugging Face
([aksenovmr/wavlm_base_unfreeze4_speech_defects](https://huggingface.co/aksenovmr/wavlm_base_unfreeze4_speech_defects))
и сохранится в папку `checkpoints/`.

### 3. Открыть интерфейс

- Streamlit-интерфейс: [http://localhost:8501](http://localhost:8501)
- API-документация: [http://localhost:8000/docs](http://localhost:8000/docs)

## Как запустить локально (без Docker)?

### 1. Клонировать репозиторий

```bash
git clone https://github.com/aksenovmr/speech-defects.git
cd speech-defects
```

### 2. Установить зависимости

```bash
pip install -r requirements.txt
```

### 3. Запустить API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### 4. Запустить Streamlit (в отдельном терминале)

```bash
streamlit run demo/streamlit_app.py
```

## Обучение модели

### Подготовить данные

Аудиозаписи разложить по папкам:

```
data/raw/хорошо/   # записи с правильным произношением
data/raw/плохо/    # записи с дефектами
```

### Запустить обучение

```bash
python scripts/train.py --config configs/train_wavlm.yaml
```

Доступные конфиги: `train_wavlm.yaml`, `train_hubert.yaml`, `train_wav2vec2.yaml`.

Логирование метрик производится через [Weights & Biases](https://wandb.ai).
Для отключения Wandb установить в конфиге:

```yaml
wandb:
  use: false
```

### Использовать свой чекпоинт

После обучения указать путь к чекпоинту в `configs/inference.yaml`:

```yaml
paths:
  checkpoint_path: checkpoints/checkpoint.pt
```


