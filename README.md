# YouTube Video Summarizer + Q&A

A Streamlit app that converts any YouTube URL into:

- chapter-by-chapter summaries
- key takeaways
- an interactive Q&A assistant

## Tech stack

- `youtube-transcript-api` for free transcript extraction
- `llama3.2:3b` via Ollama for summarization and question answering
- Streamlit for the UI

## How it works

1. Extract the YouTube video ID from the URL.
2. Pull the transcript from YouTube.
3. Split the transcript into chunks and summarize each chapter.
4. Generate key takeaways from chapter summaries.
5. Answer user questions using transcript chunks and chapter context.

## Run locally

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the app:

   ```bash
   streamlit run streamlit_app.py
   ```

3. Make sure Ollama is installed and the `llama3.2:3b` model is available.

## Notes

- No OpenAI API key is required.
- The app uses a chained pipeline: transcript extraction → chapter summaries → Q&A.
- For a single video, the transcript chunks are retrieved directly without FAISS.
