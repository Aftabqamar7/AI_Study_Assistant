
import io
import json
import re
from typing import List

import streamlit as st
from groq import Groq
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Lumina • Document AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- DARK THEME --------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg:#090d18;
    --panel:#111827;
    --line:#27334b;
    --text:#eef2ff;
    --muted:#a3afc7;
    --cyan:#67e8f9;
    --violet:#a78bfa;
}

.stApp {
    background:radial-gradient(ellipse at 12% 0%, #172346 0%, #090d18 65%);
    color:var(--text);
    font-family:'DM Sans',sans-serif;
}

h1,h2,h3 {
    font-family:'Space Grotesk',sans-serif !important;
}

.block-container {
    padding-top:1.7rem;
    max-width:1300px;
}

[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#10182a,#0b1020);
    border-right:1px solid var(--line);
}

[data-testid="stSidebar"] * {
    color:var(--text);
}

.hero {
    padding:28px 30px;
    border:1px solid #34415d;
    border-radius:22px;
    background:linear-gradient(
        120deg,
        rgba(103,232,249,.12),
        rgba(167,139,250,.12) 55%,
        rgba(17,24,39,.8)
    );
    margin-bottom:22px;
}

.eyebrow {
    text-transform:uppercase;
    letter-spacing:2px;
    color:var(--cyan);
    font-size:11px;
    font-weight:700;
}

.hero h1 {
    font-size:clamp(30px,4vw,46px);
    margin:7px 0 8px;
    line-height:1.1;
}

.hero p {
    color:#c2cce0;
    font-size:15px;
    max-width:720px;
}

.stat {
    background:rgba(21,30,48,.82);
    border:1px solid var(--line);
    padding:15px 17px;
    border-radius:15px;
    min-height:88px;
}

.stat-label {
    color:var(--muted);
    font-size:12px;
}

.stat-value {
    font-family:'Space Grotesk';
    font-size:22px;
    font-weight:700;
    margin-top:4px;
}

.panel {
    background:rgba(17,24,39,.72);
    border:1px solid var(--line);
    padding:20px;
    border-radius:18px;
}

.small-muted {
    color:var(--muted);
    font-size:13px;
}

div[data-testid="stFileUploader"] {
    border:1px dashed #465675;
    border-radius:14px;
    padding:8px;
    background:rgba(15,23,42,.5);
}

.stButton > button {
    border-radius:11px;
    border:1px solid #435274;
    background:linear-gradient(100deg,#243956,#342a5a);
    color:#fff;
    font-weight:650;
    min-height:42px;
}

.stButton > button:hover {
    border-color:var(--cyan);
    color:var(--cyan);
}

.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] {
    background:#0d1424 !important;
    color:#eef2ff !important;
    border-color:#33415d !important;
    border-radius:10px !important;
}

hr {
    border-color:var(--line);
}

div[data-testid="stTabs"] button {
    color:#bac7df;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color:var(--cyan);
}
</style>
""", unsafe_allow_html=True)


# -------------------- DOCUMENT PROCESSING --------------------

def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw))
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        )

    if name.endswith(".docx"):
        doc = Document(io.BytesIO(raw))
        parts = [
            p.text for p in doc.paragraphs if p.text.strip()
        ]

        for table in doc.tables:
            for row in table.rows:
                parts.append(
                    " | ".join(cell.text.strip() for cell in row.cells)
                )

        return "\n".join(parts)

    if name.endswith((".txt", ".md")):
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue

    raise ValueError(
        "Unsupported file type. Upload PDF, DOCX, TXT, or MD."
    )


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 180
) -> List[str]:

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(len(text), start + chunk_size)

        if end < len(text):
            boundary = max(
                text.rfind("\n", start + chunk_size // 2, end),
                text.rfind(". ", start + chunk_size // 2, end)
            )

            if boundary > start:
                end = boundary + 1

        piece = text[start:end].strip()

        if piece:
            chunks.append(piece)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def retrieve(
    question: str,
    chunks: List[str],
    top_k: int = 5
) -> List[str]:

    if not chunks:
        return []

    vectorizer = TfidfVectorizer(
        stop_words=None,
        ngram_range=(1, 2),
        max_features=50000
    )

    matrix = vectorizer.fit_transform(chunks + [question])

    scores = cosine_similarity(
        matrix[-1], matrix[:-1]
    ).ravel()

    ids = scores.argsort()[::-1][:min(top_k, len(chunks))]

    return [
        chunks[i] for i in ids if scores[i] > 0
    ]


# -------------------- GROQ AI --------------------

def call_groq(
    client,
    model,
    system,
    prompt,
    temperature=0.3
):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )

    return response.choices[0].message.content or ""


def document_prompt(context, request, language):
    return f"""Answer the user's request using the document context below.
Be accurate and helpful.

If the context does not contain the answer, clearly say that the
uploaded document does not provide enough information. Do not invent facts.

Language requested: {language}.

If Urdu is requested, answer in natural Urdu script unless the user
asks for Roman Urdu.

Explain technical words simply and use headings and bullets where useful.

DOCUMENT CONTEXT:
---
{context}
---

USER REQUEST:
{request}
"""


def parse_json_response(text):
    text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.I
    )

    match = re.search(r"\{.*\}|\[.*\]", text, flags=re.S)

    if match:
        text = match.group(0)

    return json.loads(text)


# -------------------- SESSION STATE --------------------

defaults = {
    "chunks": [],
    "doc_name": "",
    "doc_text": "",
    "chat_history": [],
    "quiz": [],
    "quiz_answers": {},
    "explanation": "",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.markdown("## ✦ Lumina")
    st.caption("Your personal document tutor")
    st.markdown("---")

    st.markdown("### 🔐 Connect to Groq")

    api_key = st.text_input(
        "Your Groq API key",
        type="password",
        placeholder="gsk_...",
        help="Enter your own Groq API key."
    )

    model = st.selectbox(
        "AI model",
        [
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-120b",
            "llama-3.1-8b-instant"
        ],
        index=0
    )

    st.caption("Get your API key at console.groq.com")

    st.markdown("---")
    st.markdown("### 📚 Add a document")

    uploaded = st.file_uploader(
        "Upload your study material",
        type=["pdf", "docx", "txt", "md"]
    )

    if st.button(
        "Analyze document",
        use_container_width=True,
        disabled=(uploaded is None)
    ):
        with st.spinner("Reading and indexing your document..."):
            try:
                extracted = extract_text(uploaded)
                chunks = chunk_text(extracted)

                if not extracted.strip() or not chunks:
                    st.error(
                        "No readable text found. Scanned PDFs may need OCR."
                    )
                else:
                    st.session_state.doc_text = extracted
                    st.session_state.chunks = chunks
                    st.session_state.doc_name = uploaded.name
                    st.session_state.chat_history = []
                    st.session_state.quiz = []
                    st.session_state.quiz_answers = {}
                    st.session_state.explanation = ""

                    st.success(
                        f"Ready! Indexed {len(chunks)} text sections."
                    )

            except Exception as e:
                st.error(f"Could not read this file: {e}")

    st.markdown("---")

    st.markdown(
        "<div class='small-muted'>"
        "Privacy tip: avoid uploading confidential documents without "
        "permission. Relevant document text and prompts are sent to "
        "Groq when you use AI features."
        "</div>",
        unsafe_allow_html=True
    )


# -------------------- MAIN HEADER --------------------

st.markdown("""
<div class="hero">
    <div class="eyebrow">READ LESS. UNDERSTAND MORE.</div>
    <h1>Turn any document into
    <span style="color:#67e8f9">understanding.</span></h1>
    <p>
        Upload a PDF or document. Get simple explanations in English
        or Urdu, ask questions grounded in your notes, and test what
        you learned with a custom quiz.
    </p>
</div>
""", unsafe_allow_html=True)


# -------------------- DASHBOARD --------------------

c1, c2, c3 = st.columns(3)

with c1:
    name = st.session_state.doc_name
    display_name = name[:25] + ("..." if len(name) > 25 else "")

    st.markdown(
        f"""<div class="stat">
        <div class="stat-label">CURRENT DOCUMENT</div>
        <div class="stat-value">{display_name if name else "Not uploaded"}</div>
        </div>""",
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f"""<div class="stat">
        <div class="stat-label">TEXT SECTIONS INDEXED</div>
        <div class="stat-value">{len(st.session_state.chunks)}</div>
        </div>""",
        unsafe_allow_html=True
    )

with c3:
    status = (
        "Ready to learn"
        if st.session_state.chunks
        else "Waiting for a file"
    )

    st.markdown(
        f"""<div class="stat">
        <div class="stat-label">STUDY MODE</div>
        <div class="stat-value">{status}</div>
        </div>""",
        unsafe_allow_html=True
    )


# -------------------- WELCOME SCREEN --------------------

if not st.session_state.chunks:
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("### ✨ What you can do")

        st.markdown("""
        <div class="panel">
            <p>🧠 <b>Simple explanations</b><br>
            <span class="small-muted">
            Turn difficult paragraphs into student-friendly language.
            </span></p>

            <p>🌐 <b>English + Urdu</b><br>
            <span class="small-muted">
            Choose the language that helps you learn best.
            </span></p>

            <p>💬 <b>Ask your document</b><br>
            <span class="small-muted">
            Get answers based on relevant sections of your file.
            </span></p>

            <p>🎯 <b>Practice quiz</b><br>
            <span class="small-muted">
            Generate multiple-choice questions and check your score.
            </span></p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("### 🚀 Get started")

        st.markdown("""
        <div class="panel">
            <ol>
                <li>Enter your Groq API key in the sidebar.</li>
                <li>Upload a PDF, DOCX, TXT, or Markdown file.</li>
                <li>Click <b>Analyze document</b>.</li>
                <li>Choose Explain, Ask, or Quiz.</li>
            </ol>

            <p class="small-muted">
            Scanned PDFs may need OCR before uploading.
            </p>
        </div>
        """, unsafe_allow_html=True)


# -------------------- STUDY WORKSPACE --------------------

if st.session_state.chunks:

    st.markdown(
        f"<div class='small-muted'>Active file: "
        f"<b>{st.session_state.doc_name}</b> · "
        f"{len(st.session_state.doc_text):,} characters extracted</div>",
        unsafe_allow_html=True
    )

    if not api_key:
        st.warning(
            "Enter your Groq API key in the sidebar to use AI features."
        )

    tabs = st.tabs([
        "✨ Explain",
        "💬 Ask questions",
        "🎯 Quiz",
        "📄 Document preview"
    ])

    if api_key:
        try:
            client = Groq(api_key=api_key)
        except Exception as e:
            st.error(f"Could not initialize Groq: {e}")
            client = None

        # -------------------- EXPLAIN --------------------

        with tabs[0]:
            st.markdown("### Make this document easy to understand")

            col1, col2 = st.columns(2)

            with col1:
                lang = st.selectbox(
                    "Explanation language",
                    [
                        "Simple English",
                        "Urdu (اردو)",
                        "Roman Urdu",
                        "English + Urdu"
                    ],
                    key="explain_lang"
                )

            with col2:
                depth = st.selectbox(
                    "Detail level",
                    [
                        "Quick summary",
                        "Topic-by-topic explanation",
                        "Detailed study notes"
                    ],
                    key="depth"
                )

            if st.button(
                "✨ Explain my document",
                use_container_width=True
            ):
                context = "\n\n".join(
                    st.session_state.chunks[
                        :min(10, len(st.session_state.chunks))
                    ]
                )

                requests = {
                    "Quick summary":
                        "Summarize the main idea, key points, and conclusion.",

                    "Topic-by-topic explanation":
                        "Explain the main topics one by one, with simple examples.",

                    "Detailed study notes":
                        "Create study notes with headings, definitions, key points, and a recap."
                }

                with st.spinner("Creating your explanation..."):
                    try:
                        result = call_groq(
                            client,
                            model,
                            "You are a patient tutor who makes complex documents easy to understand.",
                            document_prompt(
                                context,
                                requests[depth],
                                lang
                            ),
                            0.25
                        )

                        st.session_state.explanation = result

                    except Exception as e:
                        st.error(f"Groq request failed: {e}")

            if st.session_state.explanation:
                st.markdown(
                    st.session_state.explanation
                )

                st.download_button(
                    "⬇️ Download explanation (.txt)",
                    st.session_state.explanation,
                    file_name="document_explanation.txt",
                    mime="text/plain"
                )

        # -------------------- ASK QUESTIONS --------------------

        with tabs[1]:
            st.markdown("### Ask anything about your file")

            st.caption(
                "Answers use relevant sections of your document."
            )

            q_lang = st.selectbox(
                "Answer language",
                [
                    "Simple English",
                    "Urdu (اردو)",
                    "Roman Urdu",
                    "English + Urdu"
                ],
                key="qa_lang"
            )

            with st.form("ask_form", clear_on_submit=True):
                question = st.text_input(
                    "Your question",
                    placeholder="Explain the main concept with an example..."
                )

                submitted = st.form_submit_button("Ask Lumina")

            if submitted and question.strip():
                relevant = retrieve(
                    question,
                    st.session_state.chunks,
                    5
                )

                if not relevant:
                    relevant = st.session_state.chunks[:3]

                context = "\n\n---\n\n".join(relevant)

                with st.spinner("Preparing your answer..."):
                    try:
                        answer = call_groq(
                            client,
                            model,
                            "You are a helpful study assistant. Use the supplied document context and do not invent facts.",
                            document_prompt(
                                context,
                                question,
                                q_lang
                            ),
                            0.2
                        )

                        st.session_state.chat_history.append({
                            "q": question,
                            "a": answer
                        })

                    except Exception as e:
                        st.error(f"Groq request failed: {e}")

            for item in reversed(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(item["q"])

                with st.chat_message("assistant", avatar="✦"):
                    st.markdown(item["a"])

            if st.session_state.chat_history:
                if st.button("Clear conversation"):
                    st.session_state.chat_history = []
                    st.rerun()

        # -------------------- QUIZ --------------------

        with tabs[2]:
            st.markdown("### Check what you have learned")

            quiz_col1, quiz_col2, quiz_col3 = st.columns(3)

            with quiz_col1:
                quiz_count = st.select_slider(
                    "Number of questions",
                    options=[5, 8, 10, 15],
                    value=5
                )

            with quiz_col2:
                quiz_lang = st.selectbox(
                    "Quiz language",
                    [
                        "Simple English",
                        "Urdu (اردو)",
                        "Roman Urdu"
                    ],
                    key="quiz_lang"
                )

            with quiz_col3:
                quiz_level = st.selectbox(
                    "Difficulty",
                    ["Easy", "Mixed", "Challenging"],
                    index=1
                )

            if st.button(
                "🎯 Generate quiz",
                use_container_width=True
            ):
                context = "\n\n".join(
                    st.session_state.chunks[
                        :min(12, len(st.session_state.chunks))
                    ]
                )

                # FIXED: JSON braces are escaped for the f-string.
                prompt = f"""Create exactly {quiz_count} multiple-choice
questions using only the document context.

Language: {quiz_lang}.
Difficulty: {quiz_level}.

Return valid JSON only in this schema:
{{"questions":[{{"question":"...","options":["A...","B...","C...","D..."],"answer":0,"explanation":"..."}}]}}

The answer must be the zero-based index of the correct option (0-3).
Every question must have exactly four options.
Do not include Markdown code fences.

DOCUMENT CONTEXT:
{context}
"""

                with st.spinner("Building your quiz..."):
                    try:
                        raw = call_groq(
                            client,
                            model,
                            "You create accurate educational multiple-choice quizzes. Output valid JSON only.",
                            prompt,
                            0.35
                        )

                        parsed = parse_json_response(raw)

                        questions = (
                            parsed
                            if isinstance(parsed, list)
                            else parsed.get("questions", [])
                        )

                        if not questions:
                            raise ValueError(
                                "No questions were returned."
                            )

                        st.session_state.quiz = questions
                        st.session_state.quiz_answers = {}

                    except Exception as e:
                        st.error(
                            f"Could not generate the quiz. Please try again. Details: {e}"
                        )

            if st.session_state.quiz:

                with st.form("quiz_form"):
                    for i, item in enumerate(st.session_state.quiz):
                        st.markdown(
                            f"**Q{i+1}. {item.get('question', 'Question')}**"
                        )

                        options = item.get("options", [])

                        selected = st.radio(
                            "Choose one:",
                            options,
                            key=f"quiz_{i}",
                            index=None,
                            label_visibility="collapsed"
                        )

                        st.session_state.quiz_answers[i] = selected
                        st.divider()

                    submitted_quiz = st.form_submit_button(
                        "Submit answers & see score"
                    )

                if submitted_quiz:
                    score = 0

                    for i, item in enumerate(st.session_state.quiz):
                        options = item.get("options", [])
                        correct_index = item.get("answer", 0)
                        chosen = st.session_state.quiz_answers.get(i)

                        correct = (
                            options[correct_index]
                            if isinstance(correct_index, int)
                            and 0 <= correct_index < len(options)
                            else ""
                        )

                        if chosen == correct:
                            score += 1

                    total = len(st.session_state.quiz)

                    st.markdown(
                        f"## Your score: {score} / {total}"
                    )

                    st.progress(score / total if total else 0)

                    for i, item in enumerate(st.session_state.quiz):
                        options = item.get("options", [])
                        answer_index = item.get("answer", 0)

                        correct = (
                            options[answer_index]
                            if isinstance(answer_index, int)
                            and 0 <= answer_index < len(options)
                            else "Not available"
                        )

                        chosen = st.session_state.quiz_answers.get(
                            i, "No answer"
                        )

                        st.markdown(
                            f"**Q{i+1}:** "
                            f"{'✅ Correct' if chosen == correct else '❌ Incorrect'}"
                        )

                        st.markdown(
                            f"Correct answer: **{correct}**"
                        )

                        if item.get("explanation"):
                            st.caption(item["explanation"])

                st.download_button(
                    "⬇️ Download quiz as JSON",
                    json.dumps(
                        st.session_state.quiz,
                        ensure_ascii=False,
                        indent=2
                    ),
                    file_name="study_quiz.json",
                    mime="application/json"
                )

    # -------------------- DOCUMENT PREVIEW --------------------

    with tabs[3]:
        st.markdown("### Extracted text preview")

        st.caption(
            "Preview is limited to the first 12,000 characters."
        )

        st.text_area(
            "Document text",
            st.session_state.doc_text[:12000],
            height=360,
            disabled=True
        )

        st.download_button(
            "⬇️ Download extracted text (.txt)",
            st.session_state.doc_text,
            file_name="extracted_document.txt",
            mime="text/plain"
        )


# -------------------- FOOTER --------------------

st.markdown("---")

st.markdown("""
<div style="text-align:center;color:#8492ad;font-size:12px">
    ✦ LUMINA DOCUMENT AI · Built with Streamlit + Groq
</div>
""", unsafe_allow_html=True)
