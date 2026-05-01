import re
import subprocess
import textwrap
import requests
from collections import Counter
from typing import List, Dict, Optional

from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(url_or_id: str) -> str:
    url_or_id = url_or_id.strip()
    patterns = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([A-Za-z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtube\.com/embed/([A-Za-z0-9_-]{11})",
        r"(?:https?://)?youtu\.be/([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    raise ValueError("Could not parse a valid YouTube video ID from the input.")


def load_transcript(video_id: str) -> List[Dict]:
    transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en", "en-US", "en-GB"], preserve_formatting=False)
    return transcript


def _transcript_field(item, field: str):
    if isinstance(item, dict):
        return item[field]
    return getattr(item, field)


def format_transcript(transcript: List[Dict]) -> str:
    lines = []
    for item in transcript:
        start = int(_transcript_field(item, "start"))
        timestamp = f"[{start // 60:02d}:{start % 60:02d}]"
        text = _transcript_field(item, "text").replace("\n", " ")
        lines.append(f"{timestamp} {text}")
    return "\n".join(lines)


def chunk_transcript(transcript: List[Dict], max_chars: int = 2800) -> List[Dict]:
    chunks = []
    current_text = []
    current_length = 0
    current_start = int(_transcript_field(transcript[0], "start")) if transcript else 0
    current_end = current_start

    for item in transcript:
        sentence = _transcript_field(item, "text").replace("\n", " ")
        start = int(_transcript_field(item, "start"))
        token = f"[{start // 60:02d}:{start % 60:02d}] {sentence}"
        if current_length + len(token) > max_chars and current_text:
            chunks.append({
                "text": "\n".join(current_text),
                "start": current_start,
                "end": current_end,
            })
            current_text = []
            current_length = 0
            current_start = start
        current_text.append(token)
        current_length += len(token)
        current_end = start

    if current_text:
        chunks.append({
            "text": "\n".join(current_text),
            "start": current_start,
            "end": current_end,
        })
    return chunks


def _ollama_command(model: str, prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "num_predict": max_tokens,
                "stream": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama API request failed: {e}")
    except Exception as e:
        raise RuntimeError(f"Ollama generation failed: {e}")


def summarize_chunk(chunk_text: str, model: str = "llama3.2:3b") -> Dict[str, str]:
    prompt = textwrap.dedent(
        f"""
        You are a helpful assistant that summarizes one chapter of a YouTube transcript.
        Create a concise chapter title and a short chapter summary in 3-4 sentences.

        Transcript segment:
        {chunk_text}

        Output format:
        Chapter Title: <one-line title>
        Summary: <chapter summary>
        """
    )
    output = _ollama_command(model, prompt)
    title_match = re.search(r"Chapter Title:\s*(.*)", output)
    summary_match = re.search(r"Summary:\s*(.*)", output, re.DOTALL)
    return {
        "title": title_match.group(1).strip() if title_match else "Chapter summary",
        "summary": summary_match.group(1).strip() if summary_match else output.strip(),
        "raw": output.strip(),
    }


def summarize_transcript(chunks: List[Dict], model: str = "llama3.2:3b") -> List[Dict[str, str]]:
    summaries = []
    for index, chunk in enumerate(chunks, start=1):
        chapter = summarize_chunk(chunk["text"], model=model)
        chapter.update(
            {
                "chapter_index": index,
                "start": chunk["start"],
                "end": chunk["end"],
            }
        )
        summaries.append(chapter)
    return summaries


def build_takeaways(chapters: List[Dict[str, str]], model: str = "llama3.2:3b") -> Dict[str, str]:
    chapter_text = "\n\n".join(
        [f"{c['chapter_index']}. {c['title']} - {c['summary']}" for c in chapters]
    )
    prompt = textwrap.dedent(
        f"""
        You have chapter-level summaries for a YouTube video.
        1) Give me 5 concise key takeaways.
        2) Provide 5 useful follow-up questions a user might ask about the video.

        Chapter summaries:
        {chapter_text}

        Output format:
        Key Takeaways:\n- ...\n
        Suggested Questions:\n- ...
        """
    )
    output = _ollama_command(model, prompt)
    takeaways_match = re.search(r"Key Takeaways:\s*(.*?)Suggested Questions:", output, re.DOTALL)
    questions_match = re.search(r"Suggested Questions:\s*(.*)", output, re.DOTALL)
    return {
        "takeaways": takeaways_match.group(1).strip() if takeaways_match else output.strip(),
        "questions": questions_match.group(1).strip() if questions_match else "",
        "raw": output.strip(),
    }


def _score_chunk(chunk_text: str, question: str) -> int:
    question_tokens = re.findall(r"\w+", question.lower())
    chunk_tokens = re.findall(r"\w+", chunk_text.lower())
    chunk_counts = Counter(chunk_tokens)

    overlap_score = sum(chunk_counts[token] for token in question_tokens if token in chunk_counts)
    unique_overlap = len(set(question_tokens) & set(chunk_tokens))
    question_length = max(len(question_tokens), 1)
    density = overlap_score / question_length

    return int(overlap_score + 2 * unique_overlap + density * 3)


def select_context_chunks(chunks: List[Dict], question: str, top_k: int = 3) -> List[Dict]:
    scored = []
    for chunk in chunks:
        score = _score_chunk(chunk["text"], question)
        scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [chunk for score, chunk in scored[:top_k] if score > 0]
    if not selected and chunks:
        selected = [chunks[0]]
    return selected


def answer_question(question: str, transcript_chunks: List[Dict], chapter_summaries: List[Dict], model: str = "llama3.2:3b") -> str:
    context_chunks = select_context_chunks(transcript_chunks, question)
    context_text = "\n\n".join([c["text"] for c in context_chunks])
    chapters_summary = "\n\n".join([f"{c['title']}: {c['summary']}" for c in chapter_summaries])
    prompt = textwrap.dedent(
        f"""
        You are a YouTube assistant. Answer the user question using the transcript excerpts and chapter-level summarization context.
        If the answer is not in the video transcript, say "I don't have that information from the video." Keep the answer short and grounded.

        Question: {question}

        Context excerpts:
        {context_text}

        Chapter summaries:
        {chapters_summary}

        Answer:
        """
    )
    return _ollama_command(model, prompt, temperature=0.1, max_tokens=512)


def timeline_label(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"
