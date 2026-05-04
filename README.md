# YouTube Video Summarizer + Q&A

A Streamlit app that converts any YouTube URL into chapter-by-chapter summaries, key takeaways, and an interactive Q&A assistant — fully local, no API costs.

## Project Structure

```
README.md              # This file
requirements.txt       # Python dependencies
streamlit_app.py       # Streamlit UI layer
video_summarizer.py    # Core pipeline: transcript extraction, chunking, summarization, Q&A
```

## Screenshots

![App demo](assets/screenshot.png)

## Tech Stack

- `youtube-transcript-api` — free transcript extraction, no YouTube API key required
- `llama3.2:3b` via Ollama — local LLM for summarization and Q&A
- `nomic-embed-text` via Ollama — local embeddings for semantic chunk retrieval
- `numpy` — cosine similarity scoring
- Streamlit — UI layer

---

## Architecture & Design Decisions

### Pipeline overview

```
YouTube URL
    │
    ▼
extract_video_id()         # regex-based ID parsing, handles full URLs + short URLs + raw IDs
    │
    ▼
load_transcript()          # youtube-transcript-api, English variants with graceful error handling
    │
    ▼
chunk_transcript()         # character-budget chunking (~2800 chars), preserves timestamps
    │
    ▼
summarize_transcript()     # per-chunk LLM call → title + summary extraction via regex
    │
    ▼
build_takeaways()          # single LLM call over all chapter summaries → takeaways + suggested questions
    │
    ▼
Streamlit UI               # tabs: Summary / Takeaways / Q&A / Transcript
    │
    ▼ (on user question)
select_context_chunks()    # semantic retrieval via nomic-embed-text + cosine similarity
    │
    ▼
answer_question()          # LLM call grounded on top-k chunks + chapter summaries
```

### Key design decisions

**Character-budget chunking over token counting**

Chunking uses a `max_chars=2800` budget rather than token counting. This avoids a tokenizer dependency while staying well within `llama3.2:3b`'s context window. The tradeoff is slight imprecision at chunk boundaries, which is acceptable given the summarization task doesn't require exact sentence alignment.

**Per-chunk summarization, not full-transcript summarization**

The transcript is summarized chunk-by-chunk rather than feeding the full transcript in one shot. This handles videos of any length without hitting context limits, and produces chapter-level granularity that maps naturally to how people navigate video content. The cost is N sequential LLM calls instead of one — acceptable for a local model with no API cost.

**Semantic retrieval for Q&A over keyword matching**

Q&A retrieval uses `nomic-embed-text` embeddings + cosine similarity instead of keyword overlap scoring. This handles questions phrased differently from the transcript language (synonyms, paraphrasing) and is consistent with the embedding-based approach used in production RAG systems. The tradeoff is N+1 Ollama calls per question (one per chunk + one for the query) versus instant keyword scoring. For typical videos with 10-30 chunks this adds ~5-10 seconds of latency — acceptable given it runs fully locally.

**Two-stage context for Q&A**

`answer_question()` passes both the top-k raw transcript chunks and the full set of chapter summaries to the LLM. Raw chunks give precise grounded context; chapter summaries give broader video-level context for questions that span multiple sections. This reduces the chance of the model answering outside the video content.

**No FAISS for single-video use**

The RAG chatbot in this portfolio uses FAISS for persistent multi-document indexing. This project deliberately omits FAISS because the scope is a single video per session — in-memory cosine similarity over 10-30 chunks has negligible performance difference versus an indexed store, and removes a dependency. If this were extended to a multi-video library, FAISS or a vector DB would be the right addition.

**Ollama REST API over subprocess**

LLM calls use `requests.post` to Ollama's local REST API rather than shelling out via `subprocess`. This gives structured JSON responses, proper timeout control, and cleaner error propagation — more aligned with how production systems call model inference endpoints.

### Known limitations

- **No auto-caption fallback** — if English transcripts are unavailable, the app errors rather than falling back to auto-generated captions. This is a known gap.
- **Video title not fetched** — the app uses the video ID as an internal identifier. Fetching the actual title would require the YouTube Data API or HTML scraping.
- **Sequential chunk summarization** — chapters are summarized one at a time. Parallelising with `concurrent.futures` would reduce load time significantly for long videos.

---

## How It Works

1. Extract the YouTube video ID from the URL.
2. Pull the transcript from YouTube.
3. Split the transcript into character-budget chunks with timestamps preserved.
4. Summarize each chunk independently via Ollama to generate chapter titles and summaries.
5. Run a second LLM pass over all chapter summaries to extract key takeaways and suggested questions.
6. On user questions, embed the query and all chunks via `nomic-embed-text`, score by cosine similarity, retrieve top-k chunks, and generate a grounded answer.

## Run Locally

1. Activate the project virtual environment:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Start the app:

   ```powershell
   streamlit run streamlit_app.py
   ```

4. Open in browser:

   ```
   http://localhost:8501
   ```

## Ollama Setup

Ensure Ollama is installed and both models are available:

```powershell
ollama list
```

If missing, pull them:

```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

The app calls Ollama at `http://localhost:11434`.

## Notes

- No OpenAI API key required — fully local inference.
- Designed for single-video sessions; no persistent vector store.
- For multi-video or library-scale use, FAISS indexing would be the natural extension.
