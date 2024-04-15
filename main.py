from flask import Flask, request, jsonify
from gradio_client import Client

app = Flask(__name__)
client = Client("huggingface-projects/llama-2-13b-chat")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    message = data.get('message', 'Hello!!')
    system_prompt = data.get('system_prompt', 'Hello!!')
    max_tokens = data.get('max_tokens', 500)
    temperature = data.get('temperature', 0.6)
    top_p = data.get('top_p', 0.9)
    top_k = data.get('top_k', 50)
    repetition_penalty = data.get('repetition_penalty', 1.2)

    # Get the prediction result
    result = client.predict(message, system_prompt, max_tokens, temperature, top_p, top_k, repetition_penalty, api_name="/chat")

    # Print the raw response
    print("Raw Response:", result)

    # Return the result as JSON
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
