# YAPPY LCT
ВАЖНО: на git не хватило места для моделей, необходимо скачать папку по ссылке https://drive.google.com/drive/folders/1hhXSsyThiVWMYQbGmCWwFkn3e_Z8TRal?usp=sharing
 и добавить в директорию model_server.

Код решения представляет собой несколько контейнеров:
- Бот, через который происходит взаимодействие с поисковой системой;
- Инференс-сервер, в котором происходит векторизация запроса пользователя и обработка новых видео при загрузке;
- Роутер, в котором реализованы ручки для взаимодействия с векторной базой данных;
- Векторная база данных Milvus.

Для запуска системы необходимо запустить docker compose файл командой:
```bash
docker-compose up
```

После того, как все сервисы будут развернуты бот будет доступен по ссылке: [@yappy_lct_bot](https://t.me/yappy_lct_bot).

В нем реализованы возможности:
- Получить топ 5 релевантных запросу видео на любое отправленное текстовое сообщение;
- Загрузить новое видео командой `\load https://link.ru`, где вставить нужную на видео ссылку, после этого через некоторое время бот ответит, что видео успешно загружено и обработано.
