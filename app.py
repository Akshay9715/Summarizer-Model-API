from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from flask_cors import CORS

print("Imported successfully.")

app = Flask(__name__)
CORS(app)
print("App initialized...")

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model
model_name = "akshay9125/Transcript_Summerizer"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

print("Tokenizer and model loaded successfully.")


# Function to generate summary
def generate_summary(text, max_length=150):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)
    summary_ids = model.generate(inputs["input_ids"], max_length=max_length, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()

    if "transcript" not in data:
        return jsonify({"error": "Please provide a 'transcript' key in the JSON payload."}), 400

    transcript = data['transcript']
    summary = generate_summary(transcript)

    return jsonify({"summary": summary})

@app.route('/')
def home():

    return "Hello World!"

if __name__ == '__main__':
    app.run(debug=True)
