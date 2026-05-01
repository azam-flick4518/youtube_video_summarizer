import streamlit as st
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

from video_summarizer import (
    answer_question,
    build_takeaways,
    chunk_transcript,
    extract_video_id,
    format_transcript,
    load_transcript,
    summarize_transcript,
    timeline_label,
)

MODEL_NAME = "llama3.2:3b"


def init_state():
    if "transcript_chunks" not in st.session_state:
        st.session_state.transcript_chunks = []
    if "chapter_summaries" not in st.session_state:
        st.session_state.chapter_summaries = []
    if "video_title" not in st.session_state:
        st.session_state.video_title = ""
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []
    if "takeaways" not in st.session_state:
        st.session_state.takeaways = ""
    if "suggested_questions" not in st.session_state:
        st.session_state.suggested_questions = ""
    if "transcript_text" not in st.session_state:
        st.session_state.transcript_text = ""


def main():
    st.set_page_config(
        page_title="YouTube Video Summarizer + Q&A",
        page_icon="🎬",
        layout="wide",
    )
    init_state()

    st.title("YouTube Video Summarizer + Q&A")
    st.write(
        "Upload a YouTube URL, fetch the transcript, and get chapter-by-chapter summaries, key takeaways, plus an interactive Q&A mode powered by Ollama."
    )

    with st.form("video_form"):
        url = st.text_input("YouTube URL or ID", value="")
        submitted = st.form_submit_button("Load transcript & summarize")

    if submitted:
        if not url:
            st.error("Please enter a YouTube URL or video ID.")
            return
        try:
            video_id = extract_video_id(url)
            transcript = load_transcript(video_id)
            if not transcript:
                st.error("Transcript not found for this video.")
                return

            st.session_state.transcript_text = format_transcript(transcript)
            st.session_state.transcript_chunks = chunk_transcript(transcript)
            st.session_state.chapter_summaries = summarize_transcript(st.session_state.transcript_chunks, model=MODEL_NAME)
            summary_data = build_takeaways(st.session_state.chapter_summaries, model=MODEL_NAME)
            st.session_state.takeaways = summary_data["takeaways"]
            st.session_state.suggested_questions = summary_data["questions"]
            st.session_state.video_title = video_id
            st.session_state.qa_history = []
        except ValueError as exc:
            st.error(str(exc))
        except (NoTranscriptFound, TranscriptsDisabled):
            st.error("This video does not have a transcript available or it is disabled.")
        except Exception as exc:
            st.error(f"Failed to process video: {exc}")

    if st.session_state.chapter_summaries:
        st.success(f"Loaded video and generated {len(st.session_state.chapter_summaries)} chapter summaries.")

        tabs = st.tabs(["Summary", "Takeaways", "Q&A", "Transcript"])

        with tabs[0]:
            st.markdown("### Chapter-by-chapter summary")
            for chapter in st.session_state.chapter_summaries:
                label = f"{chapter['chapter_index']}. {chapter['title']} ({timeline_label(chapter['start'])} - {timeline_label(chapter['end'])})"
                with st.expander(label, expanded=False):
                    st.write(chapter["summary"])
                    if chapter.get("raw"):
                        with st.expander("View raw model output", expanded=False):
                            st.code(chapter["raw"])

        with tabs[1]:
            st.markdown("### Key takeaways")
            st.markdown(st.session_state.takeaways)
            st.markdown("### Suggested follow-up questions")
            st.markdown(st.session_state.suggested_questions)

        with tabs[2]:
            st.markdown("### Ask questions about the video")
            with st.form("qa_form"):
                question = st.text_input("Your question", value="", key="user_question")
                ask = st.form_submit_button("Ask")
            if ask:
                if not question.strip():
                    st.warning("Ask something about the video first.")
                else:
                    try:
                        answer = answer_question(question, st.session_state.transcript_chunks, st.session_state.chapter_summaries, model=MODEL_NAME)
                        st.session_state.qa_history.insert(0, {"question": question, "answer": answer})
                    except Exception as exc:
                        st.error(f"Q&A failed: {exc}")

            if st.session_state.qa_history:
                for item in st.session_state.qa_history:
                    st.markdown(f"**Q:** {item['question']}")
                    st.markdown(f"**A:** {item['answer']}")
                    st.write("---")
            else:
                st.info("Ask a question to see grounded answers based on the video transcript and chapter summaries.")

        with tabs[3]:
            st.markdown("### Transcript preview")
            with st.expander("Show transcript preview", expanded=True):
                st.code(st.session_state.transcript_text[:12000] + ("..." if len(st.session_state.transcript_text) > 12000 else ""))

        st.caption("Model: llama3.2:3b via Ollama. No OpenAI API required.")


if __name__ == "__main__":
    main()
