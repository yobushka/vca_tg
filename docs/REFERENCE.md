# REFERENCE — Контекст и подробности проекта

Этот файл содержит расширенный контекст, диагностику и технические детали по интеграции Uniview C1L-2WN (и совместимых) с Telegram через ONVIF Events.

## Аппаратный/ПО контекст
- Модель камер: Uniview C1L-2WN (HiSilicon Hi3516EV200).
- Прошивка: IPC_D1211-B0002P62D1907 (и совместимые), также встречается IMCP DIPC-B1216.9.11.250326.
- Каналы: ONVIF, RTSP, VCA-модули (Human/Motion/Line/Intrusion/Audio/Tamper).

## Диагностические наблюдения (из дампов/логов)
- Нередко NTP не настроен — скачки времени (1970-01-01) → рекомендуется включить NTP.
- SD-карта часто переполнена → включить ротацию/очистку.
- ONVIF Events: многие прошивки не реализуют GetEventServiceCapabilities — не критично.
- Подписка работает по PullPoint (CreatePullPointSubscription → PullMessages). Иногда PullPoint находится по SubscriptionReference.Address.

## Решение: vca2telegram
- Подписывается на ONVIF события у каждой камеры.
- Фильтрация событий по подстроке (topics_filter) и игнор-листу (ignore_filter) в config.yaml.
- При срабатывании отправляет в Telegram:
  - Фото (Snapshot) — HTTP, с корректной авторизацией (Basic/Digest), без user:pass в URL.
  - Видеоклип (RTSP → ffmpeg), можно указать предпочитаемый профиль потока (main/substream).
- Подпись к сообщению теперь включает:
  - Короткий триггер: [Motion/Intrusion/Tripwire/CrossLine/…].
  - Детали источника: rule=…, analytics=…, source=…, srcval=….
  - Объект: Human/Vehicle/Face (если обнаружено), количество боксов boxes=N.
- Снимок аннотируется боксами (если камера прислала координаты объекта/ROI).

## Сетевые замечания
- В docker-compose используется network_mode: host для Linux-хостов.
- На Windows/macOS host-сеть Docker Desktop не поддерживается. В этих случаях используйте bridge-сеть и убедитесь, что контейнер имеет доступ к IP камер в вашей сети (маршрутизация/файрвол).

## Замеченные форматы событий
Примеры (сокращённо):
- Motion:
  - Message … Source SimpleItem Name=VideoSourceConfigurationToken Value=video_source_config1 … Name=Rule Value=MyMotionDetectorRule
  - Data SimpleItem Name=IsMotion Value=true
- Tamper/Audio/RecordingJob/Bandwidth — встречаются и могут быть отфильтрованы через ignore_filter.

## Полезные поля
- Source.SimpleItem:
  - VideoSourceConfigurationToken, VideoAnalyticsConfigurationToken, Source, ProfileToken, …
- Data.SimpleItem:
  - IsMotion, IsTamper, IsIntrusion, Rule, …
- ElementItem/Message – могут содержать доп. XML-структуры.

## Известные ограничения/омега
- Не все камеры присылают bbox/ROI в ONVIF событиях — аннотация фото появится только при наличии координат.
- Если пароль содержит символы вроде "+", использовать HTTP Basic/Digest, не вставлять в URL (исправлено в проекте).
- Если PullPoint недоступен, возможны альтернативы (FTP upload и др.) — не включено из коробки в этой версии.

---
Дальнейшие улучшения приветствуются: добавление распознавания дополнительных типов событий, расширение парсинга bbox, интеграция альтернативных источников медиаданных.

---

## Техническое задание: Скрипт извлечения и мониторинга HumanShape / Human Detection событий

### 1. Цель
CLI/модуль для извлечения, нормализации и мониторинга событий обнаружения человека (HumanShapeAlarmOn/Off и аналоги) из офлайн PCAP, живых ONVIF PullPoint событий, пассивного sniff HTTP(S)/ONVIF трафика и файлов логов, с экспортом структурированных записей и метрик.

### 2. Источники
- PCAP (TCP 80/443 или иные указанные порты)
- Live ONVIF PullPoint (CreatePullPointSubscription + PullMessages)
- Live sniff интерфейса (BPF фильтр)
- Текстовые логи HTTP/SOAP

### 3. Ключевые сущности
- Fragment: сырое обнаруженное вхождение (On/Off JSON/XML)
- EventRecord: нормализованное событие start/end
- Episode: интервал между On и Off
- Open episode: не закрытый интервал

### 4. Вход / Параметры CLI (основные)
mode (offline_pcap|live_pullpoint|live_sniff|log_file), inputs, --camera-filter, --keywords, --time-window, --out-dir, --out-formats (csv,jsonl,metrics), --prometheus-port, --max-stream-size, --tls-decode (ключи), --state-store, --alert-webhook, --alert-throttle-seconds, --dedupe-window-seconds, --log-level.

### 5. Обработка PCAP
TCP рекассемблирование по 4‑tuple; декод: raw → gzip/deflate → base64 (эвристика); поиск JSON/kv/NotificationMessage; таймстемп из содержимого либо из пакета.

### 6. Live PullPoint
Создание подписки; цикл PullMessages (waitTime, messageLimit); парсинг NotificationMessage → выявление human по Topic либо SimpleItem (ObjectClass/TargetType/HumanShape...).

### 7. Live Sniff
Пассивный захват; тот же pipeline декодирования; ограничение памяти на поток.

### 8. Нормализация EventRecord (поля)
id, camera_ip, source_stream_id, raw_type, normalized_action (start|end), episode_id, event_timestamp (UTC ISO), ingest_timestamp, confidence?, attributes(dict), raw_fragment(hash/обрезка), transport.

### 9. Episode Logic
State machine per camera: start → открытие (dedupe_window сек для подавления дублей), end → закрытие последнего открытого; Off без On → synthetic start; авто‑закрытие по stale_timeout / max_episode_duration.

### 10. Метрики (Prometheus)
vca_events_total{camera,action}, vca_open_episodes{camera}, vca_episode_duration_seconds_sum/count, vca_processing_latency_seconds, vca_parser_errors_total{stage}, vca_bytes_processed_total, vca_streams_active.

### 11. Выходные форматы
CSV episodes (episode_id,camera,start,end,duration,fragments,source_modes), CSV events, JSONL events (полный EventRecord), HTTP /metrics, ротационный лог.

### 12. Алерты
Триггеры: новый episode, длительность > threshold, stale open; отправка в stdout / файл / webhook (JSON, backoff retries ≤3).

### 13. Конфигурация
Приоритет: CLI > ENV > config.yml. Пример блоков: cameras[], timeouts, episode (dedupe_window_seconds, stale_close_seconds).

### 14. Регулярные выражения
"?Type"?:\s*"HumanShapeAlarm(On|Off)"; HumanShapeAlarm(On|Off); NotificationMessage блок; epoch/ISO таймстемпы.

### 15. Псевдоалгоритм Offline
parse_pcap → group streams → decode_layers → extract_fragments(regex set) → normalize_event → episode_tracker.update → flush writers → finalize.

### 16. Архитектура модулей
io.pcap_loader, net.stream_reassembler, decode.layer_decoder, parse.fragment_extractor, parse.onvif_xml_parser, core.event_normalizer, core.episode_tracker, export.(csv/jsonl/metrics), alert.dispatcher, cli.entrypoint.

### 17. Edge Cases
Повтор On, Off без On, несортированные времена (clock skew), бинарный шум (>40% non-printable), декомпрессия ошибки.

### 18. Нефункциональные
Python 3.11+, 50 МБ PCAP < 60s (ориентир), потоковая память (≤ max_stream_size / поток), покрытие тестами core ≥80%.

### 19. Тесты
Юнит: On→start, On/Off→duration, Off без On, dedupe окно. Интеграция: PCAP эталон (10 fragments → 5 episodes). Нагрузка: 100k событий.

### 20. Acceptance
Обработка эталонного дампа даёт: human fragments=10, episodes=5; корректные CSV/JSONL; метрики экспонируются; повторный прогон с fingerprint-cache без дублей.

### 21. Расширения (позже)
FaceAlarmOn/Off, Kafka/MQTT, OpenTelemetry, Web UI, bounding boxes из расширенных ONVIF XML.

### 22. Ограничения
HTTPS расшифровка только без PFS (RSA ключ), возможные пропуски при фрагментации TLS.

### 23. Допущения
Epoch 10 цифр=сек, 13=мс; Off относится к последнему On той же камеры; clock skew <5s.

### 24. MVP
Offline PCAP → JSONL events + CSV episodes + базовые метрики в памяти + логирование.

### 25. План этапов
1) Data model/episode tracker
2) Regex extractor + offline pipeline
3) Writers
4) Metrics server
5) PullPoint client
6) Alerts
7) Tests/optim

---

Этот раздел зафиксирован как опорное ТЗ для реализации соответствующего скрипта.
