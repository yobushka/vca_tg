# vca2telegram — ONVIF VCA → Telegram

Минимальный сервис, который подписывается на ONVIF-события камер и отправляет в Telegram фото и короткий видеоклип при срабатывании VCA (Motion/Intrusion/Line и др.).

## Возможности
- Подписка на ONVIF Events (PullPoint) для каждой камеры.
- Фильтр по ключевым словам (topics_filter) и игнор-лист (ignore_filter).
- Подпись содержит короткий триггер и детали источника: rule=…, analytics=…, source=…, srcval=….
- Аннотация снимка: рисует bbox объекта (если камера прислала координаты).
- Клип записывается через ffmpeg из RTSP (можно выбрать сабпоток).

Подробный контекст и заметки: см. docs/REFERENCE.md

## Быстрый старт

### 1. Подготовка конфигурации
Скопируйте примеры и заполните своими данными:
```powershell
# Переменные окружения
cp .env.example .env
# Отредактируйте .env — укажите токен бота и chat_id

# Конфигурация камер
cp .config.yaml.example app/config.yaml
# Отредактируйте app/config.yaml — добавьте IP, логины, пароли камер
```

### 2. Telegram Bot Setup
1) Создайте бота через @BotFather в Telegram → получите `TELEGRAM_BOT_TOKEN`
2) Найдите chat_id:
   - Для личного чата: напишите @userinfobot — он покажет ваш ID
   - Для группы: добавьте бота в группу, затем используйте @getidsbot
   - Для канала: сделайте бота администратором канала
3) Заполните `.env`:
```bash
TELEGRAM_BOT_TOKEN=123456789:AABBCCDDxxxxxxxxxxxxxxxxxxx
TELEGRAM_CHAT_ID=-1001234567890
DEBUG=1
```

### 3. Настройка камер
Отредактируйте `app/config.yaml`:
```yaml
cameras:
  - name: entrance
    ip: 192.168.1.31
    username: admin
    password: your-camera-password
    profile: 0

topics_filter:
  - Motion          # детекция движения
  - Human           # обнаружение человека
  - Intrusion       # вторжение
  - Line            # пересечение линии
  - Cross           # кросс-детекция

ignore_filter:
  - Bandwidth       # игнорировать события изменения пропускной способности
```

### 4. Запуск и тестирование
```powershell
docker compose down -t 1
docker compose build
docker compose up
```

### 5. Ожидаемое поведение
**Нормальный запуск:**
```
vca-onvif  | Started 3 camera listeners.
vca-onvif  | [entrance] Device: UNIVIEW C1L-2WN FW IPC_D1211...
vca-onvif  | [entrance] ONVIF subscription OK. Listening for events...
```

**При срабатывании события:**
```
vca-onvif  | [entrance] Event topic: Message ... Name=IsMotion Value=true
vca-onvif  | [entrance] Event matched → sending Telegram...
```

**В Telegram вы получите:**
- Фото с боксами (если камера присылает координаты объектов)
- Подпись: `📹 entrance [Motion] @ 2025-08-11 12:34:56 rule=MyMotionDetectorRule source=video_source_config1`
- Видеоклип (10 сек, mp4)

### 6. Отладка
**Включите подробные логи:**
```bash
DEBUG=1
CAP_TEST=1  # показать возможности камеры при подключении
```

**Типичные проблемы:**
- `Skipped (no filter match)` — событие не прошло фильтр `topics_filter`
- `snapshot failed: 401` — неверный логин/пароль камеры
- `Error: Connection refused` — проверьте IP камеры и сетевую доступность
- Нет событий — проверьте настройки VCA в веб-интерфейсе камеры

**Временно отключить фильтры для диагностики:**
```yaml
topics_filter: []  # пропускать все события
```

Windows/macOS: host-сеть в Docker Desktop недоступна; текущий compose использует `network_mode: host` для Linux. При необходимости адаптируйте сеть/маршрутизацию.

## Конфигурация (`app/config.yaml`)
- cameras: список камер
  - name: произвольное имя (в подписи)
  - ip, username, password
  - profile: индекс профиля медиа (0 — по умолчанию)
- topics_filter: список ключевых слов для прохода события (case-insensitive)
- ignore_filter: список подстрок для полного игнора (например, Bandwidth)
- prefer_stream: индекс профиля для RTSP клипа (0=main, 1=sub1)

Пример:
```yaml
cameras:
  - name: cam1
    ip: 10.0.0.31
    username: admin
    password: pass

topics_filter:
  - Motion
  - Intrusion
ignore_filter:
  - Bandwidth
prefer_stream: 1
```

## Переменные окружения
- TELEGRAM_BOT_TOKEN — обязательна
- TELEGRAM_CHAT_ID — обязательна
- CLIP_SECONDS — длительность клипа (по умолчанию 10)
- COOLDOWN_SECONDS — антифлуд между отправками от одной камеры (по умолчанию 20)
- DEBUG — 1 для подробных логов

## Поведение и формат сообщений
- Заголовок: `📹 <cam> [Motion] @ 2025‑08‑11 12:34:56`.
- Детали: `rule=MyMotionDetectorRule analytics=… source=… srcval=… boxes=N`.
- Текст: первые ~180 символов исходного ONVIF-сообщения для контекста.
- Фото: snapshot через HTTP (Basic/Digest auth, без user:pass в URL).
- Видео: ffmpeg → mp4 из RTSP, предпочтительно сабпоток.

## Частые вопросы
- События Bandwidth/RecordingJob шумят — используйте `ignore_filter`.
- Motion приходит, но нет боксов — не все камеры присылают ROI/bbox в ONVIF.
- 401 на snapshot — исправлено: используем HTTP Basic/Digest, не вкладываем пароль в URL.

## Траблшутинг
- Включите DEBUG=1, смотрите строки `Event topic:` и `Skipped…` с кратким пояснением.
- Если фильтр слишком строгий, временно очистите `topics_filter:` и соберите нужные ключевые слова из логов.

## Лицензия
MIT


