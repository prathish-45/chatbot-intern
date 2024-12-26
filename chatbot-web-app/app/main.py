from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import json
import random
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os
import requests
from datetime import datetime
import pytz

app = Flask(__name__)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the intents file
def load_data():
    with open(os.path.join(os.path.dirname(__file__), 'intents.json')) as file:
        data = json.load(file)
    return data

data = load_data()
intents_list = data['intents']

# Example placeholder data (replace with actual data)
X = np.random.rand(100, 10)  # 100 samples, 10 features
labels = np.random.choice(['class_0', 'class_1'], 100)  # 100 samples, binary classification
tag_list = ['class_0', 'class_1']  # Example tag list for binary classification

# Encode labels
def encode_labels(labels, tag_list):
    label_map = {tag: idx for idx, tag in enumerate(tag_list)}
    return np.array([label_map[label] for label in labels]), label_map

y, label_map = encode_labels(labels, tag_list)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'model.pkl')

# Save the tag list
joblib.dump(tag_list, 'tag_list.pkl')

# Evaluate Model
print("Model Performance:")
y_pred = model.predict(X_test)  # Predict the labels for X_test
unique_labels = np.unique(y_test)  # Get unique classes in y_test
filtered_tag_list = [tag_list[label] for label in unique_labels]  # Filter tag_list
target_names = [tag_list[idx] for idx in unique_labels]
print(classification_report(y_test, y_pred, target_names=target_names, labels=unique_labels))

# Define the chatbot response function
def chatbot_response(user_input):
    for intent in intents_list:
        if user_input in intent['patterns']:
            if intent['tag'] == 'weather':
                return get_weather()
            elif intent['tag'] == 'time':
                return get_time()
            else:
                return random.choice(intent['responses'])
    return "I don't understand that."

def get_weather():
    api_key = "b4cc9b88af0686e5d8d557a8fe75ed3a"
    city = "Namakkal"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    print(data)  # Debugging: Print the API response
    if response.status_code == 200 and "main" in data:
        main = data["main"]
        weather_desc = data["weather"][0]["description"]
        temperature = main["temp"]
        return f"The weather in {city} is {weather_desc} with a temperature of {temperature}°C."
    else:
        return "City not found or API request failed."

def get_time():
    try:
        # Get current time in India timezone
        india_timezone = pytz.timezone('Asia/Kolkata')
        current_time = datetime.now(india_timezone)
        
        # Format time in a readable format
        formatted_time = current_time.strftime("%I:%M %p, %d %B %Y")
        return f"The current time is {formatted_time}"
    except Exception as e:
        print(f"Error in get_time(): {e}")
        return "Sorry, I couldn't fetch the current time."
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = chatbot_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)