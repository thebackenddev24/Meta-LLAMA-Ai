import os
import asyncio
from flask import Flask, request, jsonify
from gradio_client import Client
import telepot

app = Flask(__name__)

client = Client("huggingface-projects/llama-2-13b-chat")

bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
bot = telepot.Bot(bot_token)
chat_id = "-1001739431257"  # Provided chat ID

async def handle_message(msg):
    try:
        content_type, chat_type, chat_id = telepot.glance(msg)
        if content_type == 'text':
            user_message = msg['text']
            system_prompt = "null" if not user_message else user_message
            result = client.predict(system_prompt, api_name="/chat")
            bot_reply = result['generated_text'].strip()
            bot.sendMessage(chat_id, f"*Message by user*: {user_message}\n*System Prompt*: {system_prompt}\n*Bot reply*: {bot_reply}")
    except Exception as e:
        error_message = f"An error occurred while handling message: {e}"
        print(error_message)
        bot.sendMessage(chat_id, error_message)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        message = data.get('message', 'Hello!!')
        system_prompt = data.get('system_prompt', 'Hello!!')
        max_tokens = data.get('max_tokens', 500)
        temperature = data.get('temperature', 0.6)
        top_p = data.get('top_p', 0.9)
        top_k = data.get('top_k', 50)
        repetition_penalty = data.get('repetition_penalty', 1.2)
        result = client.predict(message, system_prompt, max_tokens, temperature, top_p, top_k, repetition_penalty, api_name="/chat")
        bot_reply = result['generated_text'].strip()
        bot.sendMessage(chat_id, f"Successful prediction!\n*System Prompt*: {system_prompt}\n*User Message*: {message}\n*Bot reply*: {bot_reply}")
        return jsonify(result)
    except Exception as e:
        error_message = f"An error occurred while predicting: {e}"
        print(error_message)
        bot.sendMessage(chat_id, error_message)
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    try:
        asyncio.run(bot.message_loop(handle_message))
        app.run(host='0.0.0.0', debug=True)
    except Exception as e:
        print(f"An error occurred: {e}")
