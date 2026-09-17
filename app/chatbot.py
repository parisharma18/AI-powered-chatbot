import os
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Paths
model_path = r"C:\Users\Prasoddha Aryal\Desktop\quiz-chatbot\model\fine_tuned_model"
dataset_path = r"C:\Users\Prasoddha Aryal\Desktop\quiz-chatbot\data\quiz_dataset.json"

# Load the fine-tuned classification model
classifier_model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
classifier_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

# Load the lightweight sentence embedding model (this is very fast and good)
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # small but powerful

# Load the dataset
with open(dataset_path, "r", encoding="utf-8") as f:
    quiz_data = json.load(f)

questions = [item["question"] for item in quiz_data]
answers = [item["answer"] for item in quiz_data]

# Precompute question embeddings
question_embeddings = embedder.encode(questions)

def get_answer(user_question):
    
    user_embedding = embedder.encode([user_question])[0]

    
    similarities = cosine_similarity([user_embedding], question_embeddings)[0]

   
    best_idx = np.argmax(similarities)
    best_similarity = similarities[best_idx]

    # Thresholds
    SIMILARITY_THRESHOLD = 0.7  # less strict now, realistic
    CONFIDENCE_THRESHOLD = 0.7  

    if best_similarity >= SIMILARITY_THRESHOLD:
        matched_question = questions[best_idx]
        matched_answer = answers[best_idx]

        input_text = f"Question: {user_question} Answer: {matched_answer}"
        inputs = classifier_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = classifier_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidence = probs[0][1].item()

        if confidence >= CONFIDENCE_THRESHOLD:
            return matched_answer
        else:
            return "Sorry, I don't know the answer to that question. 🤔"
    else:
        return "Sorry, I don't know the answer to that question. 🤔"

# finished