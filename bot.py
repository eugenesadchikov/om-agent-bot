import os
import logging
import asyncio
from io import BytesIO
from dotenv import load_dotenv

# --- Библиотеки для Telegram ---
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# --- Библиотеки для PDF и ИИ ---
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai

# Загружаем переменные окружения из .env файла
load_dotenv()

# ========== НАСТРОЙКИ ==========
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "ВСТАВЬТЕ_ВАШ_TELEGRAM_ТОКЕН")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "ВСТАВЬТЕ_ВАШ_GITHUB_ТОКЕН")

# Настройка клиента для GitHub Models (OpenAI-совместимый API)
client = openai.OpenAI(
    base_url="https://models.inference.ai.azure.com",  # Эндпоинт GitHub Models
    api_key=GITHUB_TOKEN,
)

# Модель для генерации ответов
MODEL_NAME = "gpt-4o"  # Можно заменить на "DeepSeek-R1" или "Llama-3-70B"

logging.getLogger("httpx").setLevel(logging.WARNING)

# Включаем логирование
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== "МОЗГ" БОТА (RAG) ==========
class KnowledgeBase:
    def __init__(self):
        # Модель для превращения текста в эмбеддинги (скачается один раз)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.index = None

    def add_document(self, text):
        """Добавляет новый текст в базу знаний"""
        if not text:
            return
        # Разбиваем текст на чанки по 500 символов
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        self.documents.extend(chunks)
        
        # Создаем эмбеддинги
        new_embeddings = self.embedder.encode(chunks)
        
        # Добавляем в индекс FAISS
        if self.index is None:
            dimension = new_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(new_embeddings.astype(np.float32))
        else:
            self.index.add(new_embeddings.astype(np.float32))
        
        logger.info(f"Добавлено {len(chunks)} чанков. Всего: {len(self.documents)}")

    def search(self, query, k=3):
        """Ищет 3 самых похожих чанка на вопрос"""
        if self.index is None or self.index.ntotal == 0:
            return []
        query_embedding = self.embedder.encode([query])
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        results = [self.documents[i] for i in indices[0] if i < len(self.documents)]
        return results

# Создаем экземпляр базы знаний
knowledge_base = KnowledgeBase()

# ========== ФУНКЦИЯ ДЛЯ ЗАПРОСА К ИИ ==========
def get_ai_response(prompt):
    """Отправляет запрос к GitHub Models и возвращает ответ"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Ты — опытный ИИ-ассистент, помогающий с алгоритмизацией задач и подготовкой решений на основе предоставленных документов. Отвечай подробно и пошагово."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Ошибка при обращении к GitHub Models: {e}")
        return None

# ========== ОБРАБОТЧИКИ КОМАНД ==========

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Приветственное сообщение"""
    await update.message.reply_text(
        "👋 Привет! Я твой личный ИИ-агент на базе GitHub Models.\n\n"
        "📄 **Отправь мне PDF-файл** с теорией, и я его изучу.\n"
        "❓ После этого задавай вопросы по материалу, и я помогу с алгоритмами и решениями.\n\n"
        f"⚙️ Используемая модель: {MODEL_NAME}"
    )

async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает полученный PDF-файл"""
    await update.message.reply_text("📥 Получил файл. Секунду, читаю и запоминаю...")
    
    # Скачиваем файл
    file = await update.message.effective_attachment.get_file()
    pdf_bytes = BytesIO()
    await file.download_to_memory(pdf_bytes)
    
    # Читаем текст из PDF
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_bytes)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        logger.error(f"Ошибка чтения PDF: {e}")
        await update.message.reply_text("❌ Не удалось прочитать PDF. Файл поврежден или защищен паролем?")
        return
    
    if not text.strip():
        await update.message.reply_text("❌ Не удалось извлечь текст из PDF. Возможно, это сканированная страница.")
        return
    
    # Добавляем текст в базу знаний
    knowledge_base.add_document(text)
    await update.message.reply_text(f"✅ Готово! Я запомнил информацию из PDF. Теперь задавай вопросы по этому материалу.")

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отвечает на текстовые вопросы пользователя, используя базу знаний"""
    user_question = update.message.text
    
    await update.message.reply_text("🤔 Думаю над вопросом...")
    
    # 1. Ищем похожие куски текста в базе знаний
    relevant_chunks = knowledge_base.search(user_question)
    
    if not relevant_chunks:
        await update.message.reply_text(
            "📚 Моя база знаний пока пуста. Отправь мне PDF-файл, чтобы я мог изучить материал и помогать тебе."
        )
        return
    
    # 2. Формируем промпт для нейросети
    context_text = "\n\n---\n\n".join(relevant_chunks)
    prompt = f"""Используй информацию из документов ниже, чтобы ответить на вопрос пользователя.
Если ответа нет в документах — скажи, что не знаешь, и предложи поискать в другом месте.
Старайся давать четкие, пошаговые инструкции.

Документы:
{context_text}

Вопрос пользователя: {user_question}

Ответ:"""
    
    # 3. Отправляем запрос в GitHub Models
    answer = get_ai_response(prompt)
    
    if answer:
        await update.message.reply_text(answer)
    else:
        await update.message.reply_text("❌ Произошла ошибка при обращении к нейросети. Попробуй еще раз через минуту.")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Ловит и логирует ошибки"""
    logger.error(f"Ошибка: {context.error}")

# ========== ЗАПУСК БОТА ==========
def main():
    """Главная функция запуска"""
    # Создаем приложение
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Добавляем обработчики
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Document.FileExtension("pdf"), handle_pdf))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))
    app.add_error_handler(error_handler)
    
    # Запускаем бота
    logger.info("🚀 Бот запущен и готов к работе!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
    