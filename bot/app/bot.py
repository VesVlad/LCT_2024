from aiogram import Bot, Dispatcher, executor, types
import os
import requests
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)

bot = Bot(token=os.getenv('BOT_TOKEN'))
dp = Dispatcher(bot)


# Стартовое сообщение бота
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    msg = '''
    Привет! Я чат-бот поиска видео по запросу.\nЧтобы получить видео, отправь свой запрос.\nЧтобы загрузить видео по ссылке введи комманду /load
    '''
    await message.answer(msg)


# Help сообщение бота
@dp.message_handler(commands=['help'])
async def help_message(message: types.Message):
    msg = '''
    '''
    await message.answer(msg)


# Ответ бота и модели на текстовый запрос пользователя
@dp.message_handler(content_types=[types.ContentType.TEXT])
async def answer(message: types.Message):

    if len(message.text) < 3:
        await message.answer('Некорректный запрос. Попробуйте еще раз.')
    if message.text.startswith('\load'):
        msg = message.text.split()[-1]
        if msg.startswith('http'):
            data = {
                'link': msg,
                'description': 'desc'
                }
            resp = requests.post('http://inference_model_server:80/upload_video', json=data)
            if resp.status_code == 200:
                await message.answer("Видео успешно загружено.")
            else:
                await message.answer("Что-то пошло не так. Попробуйте еще раз.")
        else:
            await message.answer("Неверный формат. Попробуйте еще раз.")
    else:

        data = {'query': str(message.text)}
        response = requests.post('http://inference_model_server:80/send_query', json=data)
        model_answer = response.json()['message']
        answer = "Вот видео по вашему запросу"
        await message.answer(answer)
        for i in range(len(model_answer)):
            await message.answer(f"\n{i+1}. {model_answer[i]}")


if __name__ == '__main__':
    logging.info('Starting bot')
    executor.start_polling(dp, skip_updates=True)
