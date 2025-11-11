import os
import json
import re
import whisper
import torch
import gradio as gr
from openai import OpenAI

# --- Configuration ---
MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
ERA_DEFAULT = os.getenv("DEFAULT_ERA", "1950s")
GENDER_DEFAULT = os.getenv("DEFAULT_GENDER", "female")

# Set up OpenAI client (reads OPENAI_API_KEY from environment/secrets)
client = OpenAI()

# Load Whisper model once at startup
print(f"🔄 Loading Whisper model '{MODEL_NAME}' …")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model(MODEL_NAME, device=device)
print("✅ Whisper model loaded!")

# --- Helper Functions ---

def split_into_verses(lyrics_text: str):
    """Split full lyric text into verse blocks on double-newline or large gap."""
    # Normalize newlines
    blocks = re.split(r"\n{2,}", lyrics_text.strip())
    verses = []
    for idx, block in enumerate(blocks, start=1):
        cleaned = block.strip()
        if not cleaned:
            continue
        verses.append({
            "verse": idx,
            "text": cleaned
        })
    return verses

def align_timestamps(segments, verses):
    """
    Align Whisper word-level segments to verse blocks.
    For each verse, approximate start = first segment start,
    end = last segment end before next verse begins.
    """
    aligned = []
    seg_iter = iter(segments)
    seg = next(seg_iter, None)
    for v in verses:
        start_t = None
        end_t = None
        # find first segment in this verse
        while seg and not start_t:
            start_t = seg["start"]
            end_t = seg["end"]
            seg = next(seg_iter, None)
        # then find segments until we suspect next verse (based on text match)
        # Here we simplify: just take the last found end_t
        aligned.append({
            "verse": v["verse"],
            "text": v["text"],
            "start": round(start_t, 2) if start_t is not None else None,
            "end": round(end_t, 2) if end_t is not None else None
        })
    return aligned

def detect_theme(text_block: str) -> str:
    """Call OpenAI to detect the theme of a text block (verse or full song)."""
    system_msg = {
        "role": "system",
        "content": "You are a music analyst. Summarize the main emotional or narrative theme of the following lyrics in one short sentence."
    }
    user_msg = {
        "role": "user",
        "content": text_block
    }
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[system_msg, user_msg],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def verse_image_prompt(theme: str, gender: str, era: str) -> str:
    return (
        f"{era} {gender} singer in a cinematic close-up, performing passionately in a vintage microphone scene, "
        f"visualizing the theme '{theme}'. Dramatic stage lighting, photorealistic, shallow depth of field."
    )

def instrumental_image_prompt(song_theme: str, era: str) -> str:
    return (
        f"Cinematic {era} scene symbolizing '{song_theme}' — no people visible, "
        "beautifully lit environment, photorealistic and atmospheric."
    )

# --- Main processing function ---

def process_song(audio_file_path, gender, era):
    # 1) Transcribe audio with timestamps
    print("🎧 Transcribing…")
    result = model.transcribe(audio_file_path, word_timestamps=True)
    full_lyrics = result["text"]
    segments = result["segments"]
    
    # 2) Split into verses (text blocks)
    verses = split_into_verses(full_lyrics)
    if not verses:
        # fallback: treat full lyrics as one verse
        verses = [{"verse": 1, "text": full_lyrics}]
    
    # 3) Align timestamps to each verse
    aligned = align_timestamps(segments, verses)
    
    # 4) Detect overall song theme
    print("🧠 Detecting overall song theme…")
    song_theme = detect_theme(full_lyrics)
    print(f"🎵 Detected Song Theme: {song_theme}")
    
    # 5) Build result list
    result_list = []
    for v in aligned:
        theme = detect_theme(v["text"])
        idx = v["verse"]
        prompt = verse_image_prompt(theme, gender, era)
        result_list.append({
            "start": v["start"],
            "end": v["end"],
            "type": "vocals",
            "verse": idx,
            "verse_theme": theme,
            "image_prompt": prompt
        })
        # Add an instrumental segment after each verse (optional)
        # If you prefer only verses, remove next block
        result_list.append({
            "start": v["end"],
            "end": v["end"] + 1.0,   # small gap
            "type": "instrumental",
            "verse": None,
            "verse_theme": None,
            "image_prompt": instrumental_image_prompt(song_theme, era)
        })
    
    return result_list

# --- Gradio Interface ---

def ui_generate(audio_file, gender, era):
    if audio_file is None:
        return "No file uploaded.", None
    results = process_song(audio_file, gender, era)
    return json.dumps(results, indent=2, ensure_ascii=False), results

iface = gr.Interface(
    fn=ui_generate,
    inputs=[
        gr.Audio(type="filepath", label="Upload Song"),
        gr.Radio(["female","male"], label="Singer Gender", value=GENDER_DEFAULT),
        gr.Dropdown(["1950s","1970s","1980s","1990s","2000s","modern"], label="Era", value=ERA_DEFAULT)
    ],
    outputs=[
        gr.Textbox(label="Generated Prompts JSON", lines=20),
        gr.JSON(label="Structured Output")
    ],
    title="🎵 Song → Image Prompt Generator",
    description="Upload a song. Each verse is analysed and image prompt is generated. Instrumental segments get prompts too."
)

if __name__ == "__main__":
    launch_info = iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
    if hasattr(launch_info, 'share_url') and launch_info.share_url:
        print(f"\n🚀 Public URL: {launch_info.share_url}\n")