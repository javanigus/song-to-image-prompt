import gradio as gr
import openai
import json
import re
import whisper

openai.api_key = "YOUR_OPENAI_KEY"  # Add your API key here or via Hugging Face Secrets


# ---------- Helper Functions ----------

def transcribe_audio(audio_path):
    model = whisper.load_model("small")
    result = model.transcribe(audio_path, word_timestamps=True)
    return result


def group_verses(segments, pause_threshold=2.5):
    """
    Group Whisper segments into verses using silence pauses.
    Returns list of dicts with start, end, text.
    """
    verses = []
    current = {"start": segments[0]["start"], "text": ""}
    for i, seg in enumerate(segments):
        current["text"] += " " + seg["text"].strip()
        # Look at next segment to check gap
        if i < len(segments) - 1:
            gap = segments[i + 1]["start"] - seg["end"]
            if gap > pause_threshold:
                current["end"] = seg["end"]
                verses.append(current)
                current = {"start": segments[i + 1]["start"], "text": ""}
        else:
            current["end"] = seg["end"]
            verses.append(current)
    return verses


def analyze_theme(verse_text):
    """
    Use GPT to summarize the theme of a verse.
    """
    prompt = f"""
    You are a song analyst. Summarize the main emotional theme or story of this verse
    in 2-4 words (e.g., 'romantic longing', 'heartbreak and hope', 'nostalgic love').

    Verse:
    {verse_text}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You analyze song lyrics."},
                  {"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"].strip()


def verse_prompt(verse_theme, gender="female", era="1950s"):
    """
    Generate an image prompt for a vocal verse.
    """
    return (
        f"{era} {gender} singer performing passionately in a cinematic close-up scene "
        f"that visually represents the theme '{verse_theme}'. Vintage microphone, "
        "dramatic stage lighting, photorealistic composition."
    )


def instrumental_prompt(song_theme, era="1950s"):
    """
    Generate a background/scene image prompt for instrumental parts.
    """
    return (
        f"Cinematic {era} scene symbolizing '{song_theme}' — "
        f"a beautifully lit environment evoking the song's emotional theme, "
        "no people visible, photorealistic and atmospheric."
    )


def detect_song_theme(full_lyrics):
    """
    Estimate the overall theme of the entire song.
    """
    prompt = f"""
    Analyze the overall emotional or narrative theme of this song in 3-5 words:
    ---
    {full_lyrics[:2000]}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You analyze songs and lyrics."},
                  {"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"].strip()


# ---------- Main Pipeline ----------

def process_song(audio_path, gender="female", era="1950s"):
    print("Transcribing...")
    result = transcribe_audio(audio_path)
    segments = result["segments"]
    full_text = result["text"]

    # Group verses
    verses = group_verses(segments)
    song_theme = detect_song_theme(full_text)

    structured = []

    # Generate vocal segments (verses)
    for i, v in enumerate(verses, 1):
        verse_theme = analyze_theme(v["text"])
        prompt = verse_prompt(verse_theme, gender, era)
        structured.append({
            "start": round(v["start"], 2),
            "end": round(v["end"], 2),
            "type": "vocals",
            "verse": i,
            "verse_theme": verse_theme,
            "image_prompt": prompt
        })

    # Detect instrumental gaps between verses
    instrumental_sections = []
    for i in range(len(verses) - 1):
        gap_start = verses[i]["end"]
        gap_end = verses[i + 1]["start"]
        if gap_end - gap_start > 1.5:  # Minimum silence for instrumental
            instrumental_sections.append((gap_start, gap_end))

    # Add instrumental sections to the list
    for s, e in instrumental_sections:
        structured.append({
            "start": round(s, 2),
            "end": round(e, 2),
            "type": "instrumental",
            "verse": None,
            "verse_theme": None,
            "image_prompt": instrumental_prompt(song_theme, era)
        })

    # Sort by start time
    structured = sorted(structured, key=lambda x: x["start"])
    return structured


# ---------- Gradio Interface ----------

def ui_generate(audio_file, gender, era):
    result = process_song(audio_file, gender, era)
    return json.dumps(result, indent=2)


iface = gr.Interface(
    fn=ui_generate,
    inputs=[
        gr.Audio(type="filepath", label="Upload Song"),
        gr.Radio(["female", "male"], value="female", label="Singer gender"),
        gr.Radio(["1950s", "1970s", "modern"], value="1950s", label="Era")
    ],
    outputs=gr.JSON(label="Song Structure & Image Prompts"),
    title="🎶 Song-to-Image Prompt Generator (Lyrics & Instrumentals)",
    description="Uploads a song, extracts lyrics and timestamps, determines themes per verse, and generates AI image prompts."
)

if __name__ == "__main__":
    launch_info = iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
    if hasattr(launch_info, 'share_url') and launch_info.share_url:
        print(f"\n🚀 Public URL: {launch_info.share_url}\n")


