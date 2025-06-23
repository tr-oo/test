import json #Стандартные библиотеки
import re
import logging
import os
from uuid import uuid4

import requests

from dotenv import load_dotenv # Сторонние библиотеки
from gigachat import GigaChat
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from transformers import logging as transformers_logging

load_dotenv()
GIGA_KEY=os.getenv('GIGACHAT_API_KEY')
HF_API_KEY=os.getenv("HF_API_KEY")
QDRANT_KEY=os.getenv("QDRANT_KEY")

# Настраиваем общий уровень логирования
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
# Настройка файлового хэндлера для ошибок
error_log_file = 'errors.log'
file_handler = logging.FileHandler(error_log_file) 
file_handler.setLevel(logging.ERROR) # Устанавливаем уровень (только ошибки)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Добавляем файловый хэндлер к корневому логгеру
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)

# Логгер для GigaChat. Уровень INFO позволит видеть оригинальный и очищенный текст в логах.
giga_logger = logging.getLogger("GigaChatProcessor")
giga_logger.setLevel(logging.INFO)

# Создаем логгер для общих ошибок обработки данных
data_processor_logger = logging.getLogger("DataProcessor") 
data_processor_logger.setLevel(logging.INFO) 

# Логгер для простой очистки текста
simple_clean_logger = logging.getLogger("SimpleTextCleaner")
simple_clean_logger.setLevel(logging.INFO)

# Отключаем предупреждения transformers
transformers_logging.set_verbosity_error()

# Определение GigaChat
try: 
    giga_client = GigaChat(
        credentials=GIGA_KEY, 
        verify_ssl_certs=False, 
        # scope='YOUR_SCOPE',
        # ca_bundle_file="russian_trusted_root_ca.cer"
    )
    logging.info("GigaChat клиент успешно инициализирован.")
except Exception as e:
    logging.critical(f"Критическая ошибка: Не удалось инициализировать GigaChat клиент: {e}. Программа завершает работу.")
    exit(1)

#Определение qdrant
qdrant_client = QdrantClient(
    url="https://3c72576e-1fc3-4fa5-8a64-faee22471802.europe-west3-0.gcp.cloud.qdrant.io", 
    api_key=QDRANT_KEY,
)

#Определение модели для эмбеддинга 
embedding_model = SentenceTransformer("ai-forever/ru-en-RoSBERTa")

# Простой вызов GigaChat
def my_giga():
    giga = GigaChat(
    credentials=GIGA_KEY,
    verify_ssl_certs=False,
    )

    user_prompt="Какой сейчас президент США?"

    messages_payload = {"messages": [{"role": "user", "content": user_prompt}]} #messages_payload = [{"role": "user", "content": user_prompt}] list а библиотека gigachat жидает получить объект с ключом "messages" 
    response=giga.chat(messages_payload)
    print(response.choices[0].message.content)

# Вызов GigaChat с промтом
def context_ask():
    giga = GigaChat(
    credentials=os.getenv('GIGACHAT_API_KEY'),
    verify_ssl_certs=False,
    )
    user_prompt=input("Пользователь: ")#"Расскажи график работы магазина Бристоль в городе Сыктывкар"
    
    context = """
Режим работы с 8 до 20, магазин находится на ул.Коммунистическая 52
"""
    system_promt =f"""
Ты профессиональный продавец-консультант компании. Отвечай ТОЛЬКО на основе предоставленного контекста. 
Если информации недостаточно - вежливо откажись отвечать. Сохраняй дружелюбный и уверенный тон.
**Роль:**
- Эксперт по продуктам/услугам компании
- Мастер вежливого общения
- Специалист по решению проблем клиентов

**Инструкции::**
1. Анализируй контекст из базы знаний: <CONTEXT_START>{context}<CONTEXT_END>
2. Отвечай максимально конкретно на вопрос: {user_prompt}
3. Если в контексте нет ответа: "Извините, эта информация временно недоступна. Уточните детали у менеджера"
4. Для сложных вопросов разбивай ответ на пункты
5. Избегай технического жаргона

**Стиль ответа:**
- Используй эмоции для эмоциональной окраски (1-2 в сообщении)
- Подчеркивай выгоды клиента
- Предлагай дополнительные варианты: "Возможно вас также заинтересует..."
- Завершай вопросом: "Хотите уточнить какие-то детали?"

**Пример ответа:**
"Добрый день! <основной ответ из контекста>.
Нужны дополнительные подробности? 😊"
**Технические примечания:**

- Контекст обрезан до 512 токенов
- Избегай markdown разметки
- Ответ держи в пределах 3 предложений
"""
    messages_payload = {"messages": [{"role": "user", "content": user_prompt}, {"role": "user", "content": system_promt}]} 
    response=giga.chat(messages_payload)
    print(response.choices[0].message.content)

# тестовый эмбеддинг слова
def embedding_example(text: str = "Привет", show_output: bool = True):
    """
    Генерирует векторное представление текста с помощью модели ru-en-RoSBERTa
    
    Параметры:
        text: текст для обработки (по умолчанию "Привет")
        show_output: показывать ли результат (по умолчанию True)
    
    Возвращает:
        list[float]: векторное представление текста
    """
    # Загрузка модели (кешируется после первого использования)
    model = SentenceTransformer("ai-forever/ru-en-RoSBERTa")
    
    # Преобразование текста в вектор
    embedding_vector = model.encode(text, convert_to_tensor=False).tolist()
    
    # Вывод информации
    if show_output:
        print(f"Текст: '{text}'")
        print(f"Размерность вектора: {len(embedding_vector)}")
        print(f"Первые 10 значений: {embedding_vector[:10]}")
        print("-" * 50)
    
    return embedding_vector

# Эмбеддинг локально
def get_embeddings_local(text, task = "search_document"):
    prefixed_text = f"{task}: {text}"
    #prefixed_text = text

    embedding=embedding_model.encode(
        prefixed_text,
        normalize_embeddings=True,
        convert_to_numpy=False,
        show_progress_bar=False
    ).tolist()
    #print(f"Текст для эмбеддинга (get_embeddings_local): '{prefixed_text}'")
    #print(f"Размерность вектора (get_embeddings_local): {len(embedding)}")
    #print(f"Первые 10 значений (get_embeddings_local): {embedding[:10]}")
    #print("-" * 50)                              #Првоерка работоспособности
    return embedding 

# Эмбеддинг по api
def get_embeddings_api(text ="Привет", task = "search_query"):
    API_URL = "https://router.huggingface.co/hf-inference/models/ai-forever/ru-en-RoSBERTa/pipeline/feature-extraction"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}", 
    }
    prefixed_text = f"{task}: {text}"
    #prefixed_text = text

    payload = {
        "inputs": prefixed_text,
        "parameters": {
            "pooling_method": "cls", # Явное указание пулинга
            "normalize_embeddings": True
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Создание коллекции в qdrant
def create_collections(client, collection_name, size):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=size, distance=Distance.COSINE)
    ) 

# Добавление точек в бд
def add_point(client, collection_name, text, payload):
    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct( #Запись в бд
                id=str(uuid4().hex), # Point принимает id
                vector=get_embeddings_local(text), # Сам вектор
                payload=payload
            ) 
        ])

# Чтение и запись     
def read_json_and_add_point(client, collection_name): 
    
    with open('data/RuBQ_2.0_paragraphs.json', 'r', encoding='utf-8') as file: #Вычитываем json
        data = json.load(file) # data = json.loads(file.read())

        data_processor_logger.info('Всего параграфов в файле: %s', len(data)) #Смотрим кол-во сообщений всего
        count = len(data) # Вытаскиваем кол-во сообщений

        #processed_count = 0 # тестовый запуск
        #limit = 500

        for i, entry in enumerate(data):
            #if processed_count >= limit:
                #print(f"Обработано {limit} элементов. Завершение чтения.")
                #break

            print(f'id элемента: {entry.get("uid")}')
            count -= 1
            print('Осталось:', count)

            if 'text' in entry: #Работаем с текстом
                    
                    original_text = entry.get('text')
                    text_to_embedd = simple_text_cleaning(original_text)

                    payload = { # Для фильтрации, получения и добавления в словарь
                        "content": text_to_embedd,
                        "metadata": {
                            "uid": entry.get('uid')
                        }}
                    add_point(client, collection_name, text_to_embedd, payload)
                    #processed_count += 1# тестовый запуск
            else:
                print(f"Предупреждение: Элемент на индексе {i} не содержит ключа 'text'. Пропускаем.")
        print('Завершено чтение и добавление параграфов.')

# Чтение и запись с помощью GigaChat
def read_json_and_add_point_v1(client, collection_name, giga_client):
    with open('./RuBQ_2.0_paragraphs.json', 'r', encoding='utf-8') as file: #Вычитываем json
        data = json.load(file) 

        print('всего параграфов:', len (data)) #Смотрим кол-во сообщений всего
        count = len(data) # Вытаскиваем кол-во сообщений

        processed_count = 0 # тестовый запуск
        limit = 2

        for i, entry in enumerate(data):
            if processed_count >= limit:
                print(f"Обработано {limit} элементов. Завершение чтения.")
                break
            print(f'id элемента: {entry.get("uid")}')
            count -= 1
            print('Осталось:', count)

            if 'text' in entry: #Работаем с текстом
                    # 1 способ                    
                    clear_text = entry.get('text').replace('\n', ' ').strip().lower()
                    text_to_embedd = clear_text
                    # 2 способ
                    text_to_embedd = clear_text_to_embedding(text_to_embedd, giga_client)
                    payload = { # Для фильтрации, получения и добавления в словарь
                        "content": text_to_embedd,
                        "metadata": {
                            "uid": entry.get('uid')
                        }}
                    add_point(client, collection_name, text_to_embedd, payload)

                    processed_count += 1# тестовый запуск
            else:
                print(f"Предупреждение: Элемент на индексе {i} не содержит ключа 'text'. Пропускаем.")
        print('Завершено чтение и добавление параграфов.')
 
# Чтение и запись с помощью GigaChat с чанками v2        
def read_json_and_add_point_v2(client, collection_name, giga_client):
    with open('./RuBQ_2.0_paragraphs.json', 'r', encoding='utf-8') as file: #Вычитываем json
        data = json.load(file) # data = json.loads(file.read())

        print('всего параграфов:', len (data)) #Смотрим кол-во сообщений всего
        count = len(data) # Вытаскиваем кол-во сообщений

        processed_count = 0 # тестовый запуск
        limit = 15

        for i, entry in enumerate(data):
            if processed_count >= limit:
                print(f"Обработано {limit} элементов. Завершение чтения.")
                break
            print(f'id элемента: {entry.get("uid")}')
            count -= 1
            print('Осталось:', count)

            if 'text' in entry: #Работаем с текстом
                    # 1 способ                    
                    clear_text = entry.get('text').replace('\n', ' ').strip().lower()
                    text_to_embedd = clear_text
                    # 2 способ
                    text_to_embedd = clear_text_to_embedding(text_to_embedd, giga_client)

                    text_chank = make_chanks(text_to_embedd)

                    for chank in text_chank:
                        payload = { # Для фильтрации, получения и добавления в словарь
                            "content": text_to_embedd,
                            "metadata": {
                            "uid": entry.get('uid')
                        }}
                        add_point(client, collection_name, chank, payload)

                    processed_count += 1# тестовый запуск
            else:
                print(f"Предупреждение: Элемент на индексе {i} не содержит ключа 'text'. Пропускаем.")
        print('Завершено чтение и добавление параграфов.')

# Поиск текста в бд
def find_text(client, collection_name, text):
    results = client.query_points(
        collection_name = collection_name,
        query = get_embeddings_local(text.lower()),
        limit = 5, #Количество ближайших результатов
        with_payload=True #Включить payload в результаты
    )
    for _, scope in results:
        for point in scope:
            print(f"score: {point.score}")
            print(f"text: {point.payload.get('content')}")
            print("------------")

# Очистка текста с помощью GigaChat
def clear_text_to_embedding(text: str, giga_client) -> str: # За 15 uid съел 3600 токенов

    giga_system_prompt = f"""Очисти и нормализуй текст для создания векторного представления, удаляя лишние символы и приводя к стандарту."""
    
    messages_payload= {"messages": 
        [{"role": "user", "content": giga_system_prompt},
        {"role": "user", "content": text }
        ]} 
    
    giga_logger.info("--- clear_text_to_embedding ---")
    giga_logger.info(f"Original text (first 100 chars): '{text}...'")

    try:
        response=giga_client.chat(messages_payload)

        if response and response.choices and len(response.choices) > 0 and response.choices[0].message:
            result = response.choices[0].message.content
        else:
            giga_logger.warning(f"GigaChat вернул пустой или некорректный ответ. Исходный текст: '{text[:100]}...'. Возвращаю оригинальный.")
            return text # Возвращаем исходный текст, если ответ некорректен
        
        unwanted_phrases = [
            "К сожалению, иногда генеративные языковые модели могут создавать некорректные ответы",
            "Для создания векторного представления текста можно очистить его",
            "Генеративные языковые модели не обладают собственным мнением",
            "Как и любая языковая модель, GigaChat не обладает собственным мнением",
            "Очистии нормализуйтекстдлясозданиявекторногопредставленияудаливлишниессимволыприводякстандарту" # Доп проверки качества
        ]

        for phrase in unwanted_phrases:
            if phrase.lower() in result.lower():
                giga_logger.warning(f"GigaChat вернул нежелательный шаблонный ответ. Исходный текст: '{text[:100]}...'. Возвращаю оригинальный.")
                return text # Если ответ - шаблонный мусор, возвращаем оригинальный текст
    except Exception as e:
        # Ловим любые другие ошибки, которые могут произойти во время запроса или обработки ответа
        giga_logger.error(f"Ошибка при обработке вызова или ответа API GigaChat: {e}. Исходный текст: '{text[:100]}...'. Возвращаю оригинальный.")
        return text
    
    giga_logger.info(f"Cleaned text : '{result}...'")
    return result    

# Очистка текста без GigaChat
def simple_text_cleaning(text: str) -> list[str]:
    simple_clean_logger.info("--- Simple Text Cleaning (with chunking) ---")
    simple_clean_logger.info(f"Original text : '{text}'")

    cleaned_text = text

    # 1. Удаление HTML/XML тегов
    cleaned_text = re.sub(r'<[^>]+>', ' ', cleaned_text)

    # 2. Удаление URL-адресов
    cleaned_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', cleaned_text)

    # 3. Удаление email-адресов
    cleaned_text = re.sub(r'\S*@\S*\s?', ' ', cleaned_text)

    # 4. Удаляем ВСЕ, кроме букв, цифр и пробелов
    cleaned_text = re.sub(r'[^\w\s]', ' ', cleaned_text) 

    # 5. Приведение к нижнему регистру 
    # Если вы хотите, чтобы модель различала "Москва" и "москва", не делайте tolower()
    # Если важен только смысл, можно привести.
    # cleaned_text = cleaned_text.lower() # Закомментировано, если важно сохранение регистра

    # 6. Замена множественных пробелов на одинарный пробел
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    # 7. Удаление пробелов в начале и конце строки
    cleaned_text = cleaned_text.strip()

    # 8. (Опционально) Удаление одиночных символов (если они не являются значимыми)
    # Например, если после очистки осталась одна буква 'а' или 'б'
    # cleaned_text = ' '.join([word for word in cleaned_text.split() if len(word) > 1 or word.isdigit()])

    # Проверка на пустой результат после очистки
    if not cleaned_text:
        simple_clean_logger.warning(f"Simple cleaning resulted in an empty string for text (first 100 chars): '{text[:100]}...'. Returning original text to avoid empty chunks.")
        cleaned_text = text # Возвращаем оригинальный, чтобы make_chanks получила что-то

    if cleaned_text: # Убедимся, что cleaned_text не пуст перед передачей в make_chanks
        chunks = make_chanks(cleaned_text)
        simple_clean_logger.info(f"Очищенный текст разбит на {len(chunks)} чанков. Первый чанк (first 100 chars): '{chunks[0][:100]}...'")
        return chunks
    else:
        simple_clean_logger.warning("Original text was empty and remained empty after cleaning. Returning empty list of chunks.")
        return []

# Функция для вызова создания чанков
def make_chanks(long_text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", "! ", "? ", ]
        )
    chunk = splitter.split_text(long_text)
    return chunk
    

if __name__ == "__main__":
    #my_giga() # Вызов Giga Chat 
    #context_ask() # Вызов Giga Chat с промтом
    #embedding_example() #get_embeddings_local() # Для теста в самой функции выставить text ="Привет"
    #print(qdrant_client.get_collections()) # Проверка посредством вызова коллекций в модели
       
    """ 
    COLLECTION_NAME = "my_wiki_collection" # Название библиотеки
    VECTOR_SIZE = 1024 # Размер
    create_collections(qdrant_client, COLLECTION_NAME, VECTOR_SIZE) # Создание новой коллекции
    read_json_and_add_point(qdrant_client, COLLECTION_NAME) # Добавление данных в кластер """
    
    """
    COLLECTION_NAME = "my_wiki_collection"
    read_json_and_add_point_v2(qdrant_client, COLLECTION_NAME, giga_client) #read_json_and_add_point_v2 / read_json_and_add_point_v1 / read_json_and_add_point
    """

    
    client = QdrantClient(url="https://3c72576e-1fc3-4fa5-8a64-faee22471802.europe-west3-0.gcp.cloud.qdrant.io", api_key=QDRANT_KEY,)

    text = 'про Советском Союзе'
    # text = 'Иисус Христос'
    #text = 'расскажи мне про Мисс мира'
    collection_name = 'my_wiki_collection' #Ваша бд qdrant
    find_text(client, collection_name, text)
    client.close()
   
    
   
    

#Some weights of RobertaModel were not initialized from the model checkpoint at ai-forever/ru-en-RoSBERTa and are newly initialized: ['pooler.dense.bias', 'pooler.dense.weight']
#You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.