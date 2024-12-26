# Chatbot Web Application

This project is a web application that implements a chatbot using a machine learning model. The chatbot is designed to respond to user inputs based on a trained logistic regression model.

## Project Structure

```
chatbot-web-app
├── app
│   ├── __init__.py
│   ├── main.py
│   ├── chatbot_script.py
│   └── templates
│       └── index.html
├── requirements.txt
└── README.md
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd chatbot-web-app
   ```

2. **Install dependencies:**
   Make sure you have Python installed. Then, install the required packages using pip:
   ```
   pip install -r requirements.txt
   ```

3. **Run the application:**
   Start the web server by running:
   ```
   python app/main.py
   ```

4. **Access the chatbot:**
   Open your web browser and go to `http://127.0.0.1:5000` to interact with the chatbot.

## Usage

- Type your message in the input box and press enter to receive a response from the chatbot.
- The chatbot uses a logistic regression model trained on specific data to generate responses.

## Contributing

Feel free to submit issues or pull requests if you have suggestions or improvements for the project.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.