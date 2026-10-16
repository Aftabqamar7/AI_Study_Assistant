# Lumina — RAG Document Tutor

A dark-themed Streamlit app that reads PDF, DOCX, TXT, and Markdown files and helps users study them with:
- Simple explanations in English, Urdu, or Roman Urdu
- Document-grounded Q&A using retrieval-augmented generation (RAG)
- Automatically generated multiple-choice quizzes with scoring
- Downloadable explanations, extracted text, and quiz JSON
- User-provided Groq API key (not hard-coded in the repository)

## Run locally

1. Install Python 3.10 or newer.
2. Download/unzip this project.
3. Open a terminal in the project folder and run:

   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   pip install -r requirements.txt
   streamlit run app.py
   ```

4. Open the local URL shown in the terminal.
5. Paste your own Groq API key in the sidebar. Create/manage keys at https://console.groq.com/keys.

## Deploy with GitHub + Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and `README.md`.
2. In Streamlit Community Cloud, choose the repository and `app.py` as the main file.
3. Deploy. Users enter their own Groq key in the app sidebar; no API key needs to be added to GitHub or Streamlit Secrets.

## How the RAG works

The app extracts text from uploaded documents, splits it into overlapping sections, and uses TF-IDF similarity to retrieve sections relevant to each question. The retrieved sections are sent to Groq as context for the answer. The index is built in the running app session; no vector database is required.

## Important notes

- **Scanned PDFs:** PDF text extraction cannot read text stored only as an image. Run OCR first and upload a searchable PDF.
- **Privacy:** User API keys are entered into a password field and are not written to files or saved to session state. Requests are sent to Groq, including relevant document text and prompts, when AI features are used. Do not upload sensitive documents without authorization.
- **RAG limitations:** Retrieval uses lightweight TF-IDF rather than semantic embeddings. If a document is very large or has unusual vocabulary, results may miss relevant passages.
- **Model availability:** Groq model IDs can change. If the default model is unavailable in your account, select another currently supported model in the sidebar.
- **No API key is included in this project.** Never commit personal API keys to GitHub.
