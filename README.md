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

1. Activate the project virtual environment:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

2. Install dependencies inside the virtual environment:

   ```powershell
   pip install -r requirements.txt
   ```

3. Start the app:

   ```powershell
   streamlit run streamlit_app.py
   ```

4. Open the app in the browser:

   ```text
   http://localhost:8501
   ```

## Ollama setup

- Make sure Ollama is installed locally and running.
- Ensure the model `llama3.2:3b` is available:

  ```powershell
  ollama list
  ```

- The app sends prompts to Ollama through its local REST API at `http://localhost:11434`.

## Features

- YouTube URL → transcript extraction
- chapter-by-chapter summaries
- key takeaways
- interactive Q&A using transcript retrieval and chapter context

## Notes

- No OpenAI API key is required.
- The app uses a chained pipeline: transcript extraction → chapter summaries → Q&A.
- For a single video, transcript chunks are used directly without FAISS.
