import os
import logging
import sys
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

# Загружаем переменные окружения
load_dotenv()

# ========== НАСТРОЙКИ ==========
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not TELEGRAM_BOT_TOKEN or not GITHUB_TOKEN:
    print("❌ Ошибка: не найдены токены в переменных окружения")
    sys.exit(1)

# Настройка клиента для GitHub Models
client = openai.OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_TOKEN,
)

MODEL_NAME = "gpt-4o"

# Включаем логирование
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]  # Важно для PythonAnywhere
)
logger = logging.getLogger(__name__)

# ========== БАЗА ЗНАНИЙ ==========
class KnowledgeBase:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.index = None

    def add_document(self, text):
        if not text:
            return
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        self.documents.extend(chunks)
        
        new_embeddings = self.embedder.encode(chunks)
        
        if self.index is None:
            dimension = new_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(new_embeddings.astype(np.float32))
        else:
            self.index.add(new_embeddings.astype(np.float32))
        
        logger.info(f"Добавлено {len(chunks)} чанков. Всего: {len(self.documents)}")

    def search(self, query, k=3):
        if self.index is None or self.index.ntotal == 0:
            return []
        query_embedding = self.embedder.encode([query])
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        results = [self.documents[i] for i in indices[0] if i < len(self.documents)]
        return results

knowledge_base = KnowledgeBase()

# ========== ФУНКЦИЯ ДЛЯ ЗАПРОСА К ИИ ==========
def get_ai_response(prompt):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Ты — опытный ИИ-ассистент, помогающий с алгоритмизацией задач на основе предоставленных документов."},
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
    await update.message.reply_text(
        "👋 Привет! Я твой личный ИИ-агент.\n\n"
        "📄 Отправь мне PDF-файл с теорией, и я его изучю.\n"
        "❓ После этого задавай вопросы по материалу."
    )

async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📥 Получаю файл, изучаю...")
    
    file = await update.message.effective_attachment.get_file()
    pdf_bytes = BytesIO()
    await file.download_to_memory(pdf_bytes)
    
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_bytes)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        logger.error(f"Ошибка чтения PDF: {e}")
        await update.message.reply_text("❌ Не удалось прочитать PDF.")
        return
    
    if not text.strip():
        await update.message.reply_text("❌ Не удалось извлечь текст из PDF.")
        return
    
    knowledge_base.add_document(text)
    await update.message.reply_text(f"✅ Готово! Я запомнил информацию из PDF. Теперь задавай вопросы.")

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text
    await update.message.reply_text("🤔 Думаю над вопросом...")
    
    relevant_chunks = knowledge_base.search(user_question)
    
    if not relevant_chunks:
        await update.message.reply_text(
            "📚 База знаний пуста. Сначала отправьте PDF-файл."
        )
        return
    
    context_text = "\n\n---\n\n".join(relevant_chunks)
    prompt = f"""Используй информацию из документов ниже, чтобы ответить на вопрос пользователя.
Если ответа нет в документах — скажи, что не знаешь.

Документы:
{context_text}

Вопрос: {user_question}

Ответ:"""
    
    answer = get_ai_response(prompt)
    
    if answer:
        await update.message.reply_text(answer)
    else:
        await update.message.reply_text("❌ Ошибка при обращении к нейросети. Попробуй позже.")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Ошибка: {context.error}")

# ========== ЗАПУСК БОТА ==========
def main():
    """Главная функция запуска"""
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Document.FileExtension("pdf"), handle_pdf))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))
    app.add_error_handler(error_handler)
    
    logger.info("🚀 Бот запущен и готов к работе!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
