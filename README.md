# Lumina AI Study Assistant 

Lumina is an AI-powered study app built with Streamlit and Groq.

## Features

- Upload PDF, DOCX, TXT, and Markdown files
- Ask questions about uploaded study material
- Get explanations in English, Urdu, Roman Urdu, or both
- Generate multiple-choice quizzes
- Review answers, explanations, and scores
- Modern dark interface
- Uses only `openai/gpt-oss-120b`

## Run locally

Install dependencies:

    pip install -r requirements.txt

Run the application:

    streamlit run app.py

Enter your Groq API key in the app sidebar.

## Security

Never hardcode your API key in the source code.
Never commit your API key to GitHub.
