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
