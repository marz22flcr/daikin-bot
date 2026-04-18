from dotenv import load_dotenv
import os
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()

TELEGRAM_TOKEN = ""
OPENAI_API_KEY = ""

# ── Setup OpenAI client ──────────────────────────────────────────────────────
client = OpenAI(api_key=OPENAI_API_KEY)

# ── Setup ChromaDB ───────────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings()
db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# ── AI function ──────────────────────────────────────────────────────────────
def ask_ai(question):
    docs = db.similarity_search(question, k=5)
    if not docs:
        return "Sorry, I couldn't find relevant information in the manual."

    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""You are an expert Daikin HVAC technician and manual assistant.
A customer or technician is asking you a question. Even if the question is vague or short,
do your best to give a helpful, complete answer based on the manual context below.

Rules:
- Always try to give a useful answer even for short or vague questions
- If the exact answer isn't in the context, give the closest relevant information
- If the question is about an error code, explain what it means and how to fix it
- If the question is about a component or refrigerant, explain what it is and how it works
- Use simple, clear language that a technician can understand
- Only say "I don't have that information" as a last resort if nothing is relevant at all

Context from Daikin Manual:
{context}

Question: {question}

Answer:
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert Daikin HVAC technician. Always give helpful, practical answers even for vague questions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

# ── Telegram handlers ────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *Welcome to Daikin Manual Bot!*\n\n"
        "I am an AI-powered assistant trained on the official Daikin HVAC manual. "
        "I can help technicians and users find quick, accurate answers about Daikin systems.\n\n"
        "📖 *What I can help you with:*\n"
        "• Error codes and troubleshooting\n"
        "• Refrigerant information (R32, etc.)\n"
        "• Installation and wiring guides\n"
        "• Component descriptions and functions\n"
        "• Safety procedures and warnings\n"
        "• Maintenance and inspection steps\n\n"
        "💡 *How to use:*\n"
        "Simply type your question in plain language. Examples:\n"
        "• _What does error code A3 mean?_\n"
        "• _How do I check the refrigerant level?_\n"
        "• _What is the wiring for the outdoor unit?_\n"
        "• _R32 safety precautions_\n\n"
        "⚡ Powered by OpenAI + Daikin Manual Database\n\n"
        "Go ahead, ask me anything! 🔧",
        parse_mode="Markdown"
    )
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    print(f"MESSAGE: {user_message}")
    await update.message.reply_text("⏳ Processing...")

    try:
        answer = ask_ai(user_message)
        print(f"ANSWER: {answer}")
        await update.message.reply_text(answer)
    except Exception as e:
        print(f"ERROR: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)}")

# ── Run bot ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting Daikin Manual Bot...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()