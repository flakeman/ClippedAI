"""Main script to process videos and create YouTube Shorts using AI-powered transcription and clip detection."""

import os
import pickle
import subprocess
import sys
import string
import tempfile
import contextlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nltk
from dotenv import load_dotenv
import huggingface_hub
from resize_strategies import resize_with_strategy, configure_ffmpeg

# Ensure unicode output works in Windows terminal (titles may include emojis).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Compatibility shim for libraries still passing use_auth_token.
_orig_hf_hub_download = huggingface_hub.hf_hub_download
def _hf_hub_download_compat(*args, **kwargs):
    if "use_auth_token" in kwargs and "token" not in kwargs:
        kwargs["token"] = kwargs.pop("use_auth_token")
    return _orig_hf_hub_download(*args, **kwargs)
huggingface_hub.hf_hub_download = _hf_hub_download_compat
try:
    import huggingface_hub.file_download as _hf_fd
    _hf_fd.hf_hub_download = _hf_hub_download_compat
except Exception:
    pass

# Suppress unnecessary warnings
import warnings
warnings.filterwarnings("ignore", message="Model was trained with pyannote.audio")
warnings.filterwarnings("ignore", message="Model was trained with torch")
warnings.filterwarnings("ignore", message="Lightning automatically upgraded")
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", message="torchaudio._backend.list_audio_backends has been deprecated")

# Suppress unnecessary warnings via environment variables
os.environ['FFREPORT'] = 'file=ffmpeg.log:level=32'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TQDM_DISABLE'] = '1'

# Load environment variables
load_dotenv()
FFMPEG_EXE, FFPROBE_EXE = configure_ffmpeg()
NLTK_AUTO_DOWNLOAD = os.getenv("NLTK_AUTO_DOWNLOAD", "false").lower() == "true"

# Some third-party libraries call nltk.download() unconditionally.
# In offline runs this creates noisy network errors, so disable by default.
_orig_nltk_download = nltk.download
def _safe_nltk_download(*args, **kwargs):
    if not NLTK_AUTO_DOWNLOAD:
        return False
    kwargs.setdefault("quiet", True)
    try:
        return _orig_nltk_download(*args, **kwargs)
    except Exception:
        return False
nltk.download = _safe_nltk_download

from clipsai import Transcriber, ClipFinder
from clipsai.clip.clip import Clip

def ensure_nltk_data(resource: str) -> None:
    """Ensure NLTK resource exists; optional quiet download if missing."""
    try:
        nltk.data.find(f"tokenizers/{resource}")
        return
    except LookupError:
        pass
    if not NLTK_AUTO_DOWNLOAD:
        print(f"Warning: NLTK resource '{resource}' not found. Set NLTK_AUTO_DOWNLOAD=true to fetch it.")
        return
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        print(f"Warning: NLTK resource '{resource}' is not available.")


ensure_nltk_data("punkt")
ensure_nltk_data("punkt_tab")
if not FFMPEG_EXE or not FFPROBE_EXE:
    print(
        "Warning: ffmpeg/ffprobe not found. Set FFMPEG_EXE and FFPROBE_EXE in .env "
        "or install ffmpeg in PATH. Video trim/resize may fail."
    )

# --- Directories ---
INPUT_DIR = os.getenv("INPUT_DIR", "input")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
FONT_DIR = "fonts" 

# --- Hugging Face ---
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "your_huggingface_token_here")
if HUGGINGFACE_TOKEN and HUGGINGFACE_TOKEN != "your_huggingface_token_here":
    # Some libraries read HF_TOKEN instead of custom variable names.
    os.environ["HF_TOKEN"] = HUGGINGFACE_TOKEN

# --- Clip Duration ---
MIN_CLIP_DURATION = int(os.getenv("MIN_CLIP_DURATION", "45"))
MAX_CLIP_DURATION = int(os.getenv("MAX_CLIP_DURATION", "120"))

# --- Transcription ---
TRANSCRIPTION_MODEL = os.getenv("TRANSCRIPTION_MODEL", "large-v1")
MAX_CLIPS_OVERRIDE = int(os.getenv("MAX_CLIPS_OVERRIDE", "0"))

# --- Resizing ---
ASPECT_RATIO_WIDTH = int(os.getenv("ASPECT_RATIO_WIDTH", "9"))
ASPECT_RATIO_HEIGHT = int(os.getenv("ASPECT_RATIO_HEIGHT", "16"))
RESIZE_MODE = os.getenv("RESIZE_MODE", "auto")

# --- Groq API ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
AI_RESIZE_ALLOWED = True
SHOW_TITLE_PROMPT = os.getenv("SHOW_TITLE_PROMPT", "true").lower() == "true"
QUIET_MODE = os.getenv("QUIET_MODE", "true").lower() == "true"
TITLE_PROMPT_TEMPLATE = os.getenv(
    "TITLE_PROMPT_TEMPLATE",
    "Given the following transcript, generate a catchy, viral YouTube Shorts title (max 7 words). "
    "ALWAYS include an emoji in the title. ONLY output the title, nothing else. Do NOT use hashtags. "
    "Do NOT explain, do NOT repeat the prompt, do NOT add quotes. The title should be in the style of these examples: "
    "{examples}.\n\nTranscript:\n{transcript}"
)

# --- Subtitles ---
SUBTITLE_FONT = "Montserrat Extra Bold"
SUBTITLE_FONT_SIZE = 80
SUBTITLE_ALIGNMENT = 8  # Top center
SUBTITLE_MARGIN_V = 120
SUBTITLE_STYLES = {
    "Default": {
        "Fontname": SUBTITLE_FONT,
        "Fontsize": SUBTITLE_FONT_SIZE,
        "PrimaryColour": "&H00FFFFFF",  # White
        "SecondaryColour": "&H000000FF",
        "OutlineColour": "&H40000000",
        "BackColour": "&HFF000000",
        "Bold": -1,
        "Italic": 0,
        "Underline": 0,
        "StrikeOut": 0,
        "ScaleX": 100,
        "ScaleY": 100,
        "Spacing": 2,
        "Angle": 0,
        "BorderStyle": 1,
        "Outline": 15,
        "Shadow": 0,
        "Alignment": SUBTITLE_ALIGNMENT,
        "MarginL": 30,
        "MarginR": 30,
        "MarginV": SUBTITLE_MARGIN_V,
        "Encoding": 1,
    },
    "Yellow": {
        "Fontname": SUBTITLE_FONT,
        "Fontsize": SUBTITLE_FONT_SIZE,
        "PrimaryColour": "&H0000FFFF",  # Yellow
        "SecondaryColour": "&H000000FF",
        "OutlineColour": "&H40000000",
        "BackColour": "&HFF000000",
        "Bold": -1,
        "Italic": 0,
        "Underline": 0,
        "StrikeOut": 0,
        "ScaleX": 100,
        "ScaleY": 100,
        "Spacing": 2,
        "Angle": 0,
        "BorderStyle": 1,
        "Outline": 15,
        "Shadow": 0,
        "Alignment": SUBTITLE_ALIGNMENT,
        "MarginL": 30,
        "MarginR": 30,
        "MarginV": SUBTITLE_MARGIN_V,
        "Encoding": 1,
    },
}

def get_transcription_file_path(input_path: str) -> str:
    """Generate the transcription file path based on input video path."""
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(INPUT_DIR, f"{base_name}_transcription.pkl")

def load_existing_transcription(transcription_path: str):
    """Load existing transcription if it exists."""
    if os.path.exists(transcription_path):
        print(f"Found existing transcription: {transcription_path}")
        try:
            with open(transcription_path, "rb") as f:
                transcription = pickle.load(f)
            print("Successfully loaded existing transcription!")
            return transcription
        except Exception as e:
            print(f"Error loading existing transcription: {e}")
            return None
    return None

def save_transcription(transcription, transcription_path: str):
    """Save transcription to file."""
    try:
        with open(transcription_path, "wb") as f:
            pickle.dump(transcription, f)
        print(f"Transcription saved to: {transcription_path}")
    except Exception as e:
        print(f"Error saving transcription: {e}")

def ass_time(seconds: float) -> str:
    """Convert seconds to ASS time format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centisecs:02d}"

def safe_filename(s: str) -> str:
    """Remove characters not allowed in filenames, but keep spaces, punctuation, and emojis."""
    # Define a set of characters that are generally safe for filenames
    # This includes alphanumeric, spaces, and some common punctuation
    safe_chars = string.ascii_letters + string.digits + " -_."
    # Add common punctuation that might be part of a title but needs careful handling
    safe_chars += "!?,:;@#$%^&+=[]{}"
    # Add a range of common emojis. This list can be expanded if needed.
    # Using a broad range to cover most common emojis without making the string excessively long.
    emoji_chars = "".join(chr(i) for i in range(0x1F600, 0x1F64F)) + \
                  "".join(chr(i) for i in range(0x1F300, 0x1F5FF)) + \
                  "".join(chr(i) for i in range(0x1F900, 0x1F9FF)) + \
                  "".join(chr(i) for i in range(0x1FA70, 0x1FAFF))
    
    valid_chars = safe_chars + emoji_chars
    return ''.join(c for c in s if c in valid_chars)


def run_quietly(func, *args, **kwargs):
    """Run noisy third-party calls with suppressed stdout/stderr in quiet mode."""
    if not QUIET_MODE:
        return func(*args, **kwargs)
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            return func(*args, **kwargs)


def get_font_path(font_name: str) -> str:
    """Get the path to a font file in the font directory."""
    # Try different extensions for the font file
    for ext in ['.ttf', '.otf', '.TTF', '.OTF']:
        font_path = os.path.join("fonts", f"{font_name}{ext}")
        if os.path.exists(font_path):
            return font_path
    # If not found with extension, return as is (for system fonts)
    return os.path.join("fonts", font_name)

def transcribe_with_progress(audio_file_path, transcriber):
    """Transcribe with progress tracking"""
    print('Transcribing video...')
    
    # Get video duration for progress calculation
    try:
        probe_cmd = [FFPROBE_EXE, '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', audio_file_path]
        duration = float(subprocess.check_output(probe_cmd).decode().strip())
        print(f"Video duration: {duration:.2f} seconds")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        duration = 0
        print(f"Could not determine video duration for progress tracking: {e}")
    
    # Custom progress callback
    def progress_callback(current_time):
        if duration > 0:
            progress = (current_time / duration) * 100
            print(f"Transcription progress: {progress:.1f}% ({current_time:.1f}s / {duration:.1f}s)") 
        else:
            print(f"Transcription progress: {current_time:.1f}s processed")
    
    # For now, we'll use a simple approach since clipsai doesn't expose progress directly
    # You can enhance this by modifying the clipsai library or using a different approach
    print("Starting transcription (progress updates may be limited)...")
    temp_wav_path = None
    try:
        # Fail fast when a video has no audio stream.
        probe_audio_cmd = [
            FFPROBE_EXE, "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            audio_file_path,
        ]
        audio_streams = subprocess.check_output(probe_audio_cmd, text=True).strip()
        if not audio_streams:
            print("No audio stream found in the video. Skipping this video.")
            return None

        # Work around clipsai/whisperx mp4 handling issues by forcing a wav input.
        fd, temp_wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        extract_cmd = [
            FFMPEG_EXE, "-y", "-i", audio_file_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            temp_wav_path
        ]
        subprocess.run(extract_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        transcription = run_quietly(
            transcriber.transcribe,
            audio_file_path=temp_wav_path,
            iso6391_lang_code='en',
        )
    except IndexError:
        # whisperx may raise IndexError when VAD returns no speech segments.
        print("No active speech detected. Skipping this video.")
        return None
    except Exception as e:
        print(f"Transcription failed: {e}")
        return None
    finally:
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except Exception:
                pass
    print("Transcription completed!")
    return transcription


def trim_video_ffmpeg(
    input_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
) -> bool:
    """Trim clip with ffmpeg instead of clipsai MediaEditor."""
    duration = max(0.0, end_time - start_time)
    if duration <= 0:
        return False
    cmd = [
        FFMPEG_EXE,
        "-y",
        "-ss",
        f"{start_time:.3f}",
        "-i",
        input_path,
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        print(
            "Trim failed: ffmpeg executable not found. "
            "Configure FFMPEG_EXE/FFPROBE_EXE or install ffmpeg in PATH."
        )
        return False
    except Exception:
        return False


def create_animated_subtitles(video_path, transcription, clip, output_path):
    """
    Create clean, bold subtitles matching the provided style: white bold for text, yellow bold for numbers/currency, no effects, TOP CENTER.
    """
    print('Creating styled subtitles...')
    
    # Get word info for the clip
    # Keep words that overlap clip boundaries to avoid losing words at clip edges.
    word_info = [
        w for w in transcription.get_word_info()
        if w["end_time"] > clip.start_time and w["start_time"] < clip.end_time
    ]
    if not word_info:
        print('No word-level transcript found for the clip. Skipping subtitles.')
        return video_path

    # Build cues: group words into phrases of max 25 chars
    cues = []
    current_cue = {
        'words': [],
        'start_time': None,
        'end_time': None
    }
    
    for w in word_info:
        word = w["word"]
        start_time = max(0.0, w["start_time"] - clip.start_time)
        end_time = min(clip.end_time - clip.start_time, w["end_time"] - clip.start_time)
        
        should_start_new = False
        if current_cue['start_time'] is None:
            should_start_new = True
        elif len(' '.join(current_cue['words']) + ' ' + word) > 25:
            should_start_new = True
        elif start_time - current_cue['end_time'] > 0.5:
            should_start_new = True
        
        if should_start_new:
            if current_cue['words']:
                cues.append({
                    'start': current_cue['start_time'],
                    'end': current_cue['end_time'],
                    'text': ' '.join(current_cue['words'])
                })
            current_cue = {
                'words': [word],
                'start_time': start_time,
                'end_time': end_time
            }
        else:
            current_cue['words'].append(word)
            current_cue['end_time'] = end_time
    if current_cue['words']:
        cues.append({
            'start': current_cue['start_time'],
            'end': current_cue['end_time'],
            'text': ' '.join(current_cue['words'])
        })
    
    # Determine font used and print to console
    font_used = "Montserrat Extra Bold"
    print(f"Subtitles will use font: {font_used}")
    print("NOTE: Ensure 'Montserrat Extra Bold' font is installed or is available in the fonts directory.")

    # Write ASS subtitle file with clean, bold styling at the TOP CENTER
    ass_file = os.path.join(OUTPUT_DIR, 'temp_subtitles.ass')
    with open(ass_file, 'w', encoding='utf-8') as f:
        f.write("""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 1
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Montserrat Extra Bold,80,&H00FFFFFF,&H000000FF,&H40000000,&HFF000000,-1,0,0,0,100,100,2,0,1,15,0,8,30,30,120,1
Style: Yellow,Montserrat Extra Bold,80,&H0000FFFF,&H000000FF,&H40000000,&HFF000000,-1,0,0,0,100,100,2,0,1,15,0,8,30,30,120,1
Style: Fallback,Arial Rounded MT Bold,80,&H00FFFFFF,&H000000FF,&H40000000,&HFF000000,-1,0,0,0,100,100,2,0,1,15,0,8,30,30,120,1
Style: FallbackYellow,Arial Rounded MT Bold,80,&H0000FFFF,&H000000FF,&H40000000,&HFF000000,-1,0,0,0,100,100,2,0,1,15,0,8,30,30,120,1
Style: Fallback2,Arial Black,80,&H00FFFFFF,&H000000FF,&H40000000,&HFF000000,-1,0,0,0,100,100,2,0,1,15,0,8,30,30,120,1
Style: Fallback2Yellow,Arial Black,80,&H0000FFFF,&H000000FF,&H40000000,&HFF000000,-1,0,0,0,100,100,2,0,1,15,0,8,30,30,120,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""")
        for cue in cues:
            start = ass_time(cue['start'])
            end = ass_time(cue['end'])
            words = cue['text'].split()
            line = ''
            for w in words:
                if any(char.isdigit() for char in w) or '$' in w or (',' in w and w.replace(',', '').isdigit()):
                    line += f'{{\\style Yellow}}{w} '
                else:
                    line += f'{w} '
            line = line.strip()
            f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{line}\n")
    
    final_output = output_path.replace('.mp4', '_with_subtitles.mp4')
    # ffmpeg ass filter on Windows needs escaped drive colon and forward slashes.
    ass_file_filter_path = ass_file.replace("\\", "/").replace(":", "\\:")
    ffmpeg_cmd = [
        FFMPEG_EXE, '-i', video_path,
        '-vf', f"ass=filename='{ass_file_filter_path}'",
        '-c:a', 'copy',
        '-y',
        final_output
    ]
    try:
        result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        os.remove(ass_file)
        print(f'Styled subtitles added successfully!')
        return final_output
    except subprocess.CalledProcessError as e:
        print(f'Error adding subtitles: {e}')
        print(f'FFmpeg stderr: {e.stderr.decode()}')
        print(f'FFmpeg stdout: {e.stdout.decode()}')
        return video_path

def get_viral_title(transcript_text, groq_api_key):
    import requests
    # Limit examples to avoid too long prompt
    examples = [
        "She was almost dead 😵", "He made $1,000,000 in 1 hour 💸", "This changed everything... 😲", 
        "They couldn't believe what happened! 😱", "He risked it all for this 😬"
    ]
    prompt = TITLE_PROMPT_TEMPLATE.format(
        examples=", ".join(examples),
        transcript=transcript_text
    )
    if SHOW_TITLE_PROMPT:
        print("\n[Title Prompt]")
        print(prompt[:1200] + ("..." if len(prompt) > 1200 else ""))
    headers = {
        'Authorization': f'Bearer {groq_api_key}',
        'Content-Type': 'application/json',
    }
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 30,
        "temperature": 0.7,
        "top_p": 0.9
    }
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        result = response.json()
        # Just return the first line of the response as the title, and filter out any lines that look like explanations or quotes
        content = result['choices'][0]['message']['content']
        lines = [l.strip('"') for l in content.strip().split('\n') if l.strip() and not l.lower().startswith('here') and not l.lower().startswith('title:')]
        title = lines[0] if lines else "Untitled Clip"
        return title
    except requests.exceptions.HTTPError as e:
        print(f"Error with Groq API: {e}")
        print(f"Response status code: {response.status_code}")
        print(f"Response text: {response.text}")
        return "Untitled Clip"
    except Exception as e:
        print(f"Unexpected error with Groq API: {e}")
        return "Untitled Clip"

def calculate_engagement_score(clip, transcription):
    """
    Calculate a custom engagement score for a clip based on available data.
    Higher scores indicate more engaging content.
    """
    # Get words in the clip
    clip_words = [w for w in transcription.get_word_info() 
                  if w["start_time"] >= clip.start_time and w["end_time"] <= clip.end_time]
    
    if not clip_words:
        return 0.0
    
    # Calculate various engagement factors
    duration = clip.end_time - clip.start_time
    word_count = len(clip_words)
    word_density = word_count / duration if duration > 0 else 0
    
    # Count numbers, currency, and exclamation marks (engagement indicators)
    engagement_words = 0
    for word_info in clip_words:
        word = word_info["word"]
        if any(char.isdigit() for char in word) or '$' in word or '!' in word:
            engagement_words += 1
    
    # Calculate engagement score (0-1 scale)
    # Factors: word density (45%), engagement words ratio (30%), duration balance (25%)
    word_density_score = min(word_density / 3.0, 1.0)  # Normalize to 0-1
    engagement_ratio = engagement_words / word_count if word_count > 0 else 0
    duration_score = min(duration / 75.0, 1.0)  # Prefer clips around 75 seconds
    
    engagement_score = (word_density_score * 0.45 + 
                       engagement_ratio * 0.30 + 
                       duration_score * 0.25)
    
    return engagement_score


def fallback_find_clips_from_words(
    transcription,
    min_duration: int,
    max_duration: int,
    max_candidates: int = 12,
) -> List[Clip]:
    """Offline-safe fallback clip builder using transcript word timing."""
    words = transcription.get_word_info() or []
    words = [w for w in words if "start_time" in w and "end_time" in w]
    if not words:
        return []
    words.sort(key=lambda w: w["start_time"])

    target = max(min_duration, min(max_duration, 65))
    clips: List[Clip] = []
    i = 0
    n = len(words)
    while i < n and len(clips) < max_candidates:
        start = float(words[i]["start_time"])
        min_end = start + min_duration
        target_end = start + target
        max_end = start + max_duration

        end = None
        j = i
        while j < n and float(words[j]["end_time"]) <= max_end:
            cur_end = float(words[j]["end_time"])
            if cur_end >= min_end:
                end = cur_end
                if cur_end >= target_end:
                    break
            j += 1

        if end is not None and end > start:
            clips.append(Clip(start_time=start, end_time=end, start_char=0, end_char=0))
            # Move forward to avoid creating highly overlapping clips.
            while i < n and float(words[i]["start_time"]) < start + (target * 0.6):
                i += 1
        else:
            i += 1
    return clips

# Find all mp4 files in the input directory
input_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.mp4')]
if not input_files:
    raise FileNotFoundError('No mp4 file found in input directory.')

# Find all transcription files in the input directory
transcription_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('_transcription.pkl')]

# If more than one mp4, ask user to match transcription files (if any)
video_transcription_map = {}
if len(input_files) > 1:
    print("Multiple video files detected:")
    for idx, f in enumerate(input_files, 1):
        print(f"  {idx}) {f}")
    print("\nAvailable transcription files:")
    for idx, f in enumerate(transcription_files, 1):
        print(f"  {idx}) {f}")
    print("\nFor each video, enter the number of the matching transcription file, or 0 to transcribe from scratch.")
    for vid_idx, video_file in enumerate(input_files, 1):
        while True:
            try:
                match = input(f"Match transcription for '{video_file}' (0 for none): ").strip().replace('\n', '')
                match_idx = int(match)
                if match_idx == 0:
                    video_transcription_map[video_file] = None
                    break
                elif 1 <= match_idx <= len(transcription_files):
                    video_transcription_map[video_file] = transcription_files[match_idx-1]
                    break
                else:
                    print("Invalid choice. Try again.")
            except Exception:
                print("Invalid input. Try again.")
else:
    # Only one video, try to auto-match
    video_file = input_files[0]
    base_name = os.path.splitext(os.path.basename(video_file))[0]
    expected_trans = f"{base_name}_transcription.pkl"
    if expected_trans in transcription_files:
        video_transcription_map[video_file] = expected_trans
    else:
        video_transcription_map[video_file] = None

# Prompt user for number of clips for each video BEFORE any processing
video_max_clips = {}
clip_ranges = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12)]
for video_file in video_transcription_map:
    if MAX_CLIPS_OVERRIDE > 0:
        max_clips = MAX_CLIPS_OVERRIDE
        print(f"\nUsing MAX_CLIPS_OVERRIDE={max_clips} for '{video_file}'.")
    else:
        print(f"\nHow many clips do you want for '{video_file}'?")
        for i, (low, high) in enumerate(clip_ranges, 1):
            print(f"  {i}) {low}-{high}")
        try:
            user_choice = int(input("Your choice: ").strip().replace('\n', ''))
            if not (1 <= user_choice <= len(clip_ranges)):
                raise ValueError
        except Exception:
            print("Invalid input. Defaulting to 2 clips.")
            user_choice = 1
        max_clips = clip_ranges[user_choice-1][1]
    print(f"Will select up to {max_clips} clips (if available and engaging).\n")
    video_max_clips[video_file] = max_clips

# Process each video file
processed_videos = 0
skipped_videos = 0
for video_idx, (video_file, transcription_file) in enumerate(video_transcription_map.items(), 1):
    print(f"\n=== Processing Video {video_idx}/{len(video_transcription_map)}: {video_file} ===")
    input_path = os.path.abspath(os.path.join(INPUT_DIR, video_file))
    transcription_path = os.path.join(INPUT_DIR, transcription_file) if transcription_file else get_transcription_file_path(input_path)
    max_clips = video_max_clips[video_file]

    # 1. Transcribe the video (or load existing)
    transcriber = Transcriber(model_size=os.getenv('TRANSCRIPTION_MODEL', 'large-v1'))
    transcription = load_existing_transcription(transcription_path) if transcription_file else None
    if transcription is None:
        transcription = transcribe_with_progress(input_path, transcriber)
        if transcription is None:
            print(f"Skipping '{video_file}' due to transcription failure or no speech.")
            skipped_videos += 1
            continue
        save_transcription(transcription, transcription_path)

    # 2. Find clips
    clipfinder = ClipFinder()
    try:
        clips = run_quietly(clipfinder.find_clips, transcription=transcription)
    except Exception as e:
        print(f"ClipFinder failed ({e}). Using offline fallback clip selection.")
        clips = fallback_find_clips_from_words(
            transcription=transcription,
            min_duration=MIN_CLIP_DURATION,
            max_duration=MAX_CLIP_DURATION,
            max_candidates=max(12, max_clips * 3),
        )
    if not clips:
        print('No clips found in the video.')
        continue

    # 3. Filter clips by duration and select the best ones
    valid_clips = [c for c in clips if MIN_CLIP_DURATION <= (c.end_time - c.start_time) <= MAX_CLIP_DURATION]
    selected_clips = []

    if valid_clips:
        # Calculate engagement scores for all valid clips
        clip_scores = [(clip, calculate_engagement_score(clip, transcription)) for clip in valid_clips]
        # Sort by engagement score (highest first)
        clip_scores.sort(key=lambda x: x[1], reverse=True)
        # Select up to max_clips, but only include clips with engagement >= 0.6 (for 3rd and beyond)
        for i, (clip, score) in enumerate(clip_scores):
            if i < 2 or score >= 0.6:
                if len(selected_clips) < max_clips:
                    selected_clips.append(clip)
            else:
                break
        print(f'Selected top {len(selected_clips)} clips:')
        for i, clip in enumerate(selected_clips):
            score = calculate_engagement_score(clip, transcription)
            print(f'  Clip {i+1}: {clip.start_time:.1f}s - {clip.end_time:.1f}s (duration: {clip.end_time - clip.start_time:.1f}s, engagement: {score:.3f})')
        print(f'Clip selection criteria: Top engaging clips within {MIN_CLIP_DURATION}-{MAX_CLIP_DURATION} second range')
    else:
        print(f'No clips found between {MIN_CLIP_DURATION} and {MAX_CLIP_DURATION} seconds.')
        # Find clips that are too short and try to extend them
        short_clips = [c for c in clips if c.end_time - c.start_time < MIN_CLIP_DURATION]
        if short_clips:
            print('Attempting to extend most engaging short clips to minimum duration...')
            short_clip_scores = [(clip, calculate_engagement_score(clip, transcription)) for clip in short_clips]
            short_clip_scores.sort(key=lambda x: x[1], reverse=True)
            # Take top 2 short clips and extend them
            for i, (clip, score) in enumerate(short_clip_scores[:2]):
                if clip.end_time - clip.start_time < MIN_CLIP_DURATION:
                    extension_needed = MIN_CLIP_DURATION - (clip.end_time - clip.start_time)
                    max_extension = min(extension_needed, MAX_CLIP_DURATION - (clip.end_time - clip.start_time))
                    extended_clip = Clip(
                        start_time=clip.start_time,
                        end_time=clip.end_time + max_extension,
                        start_char=clip.start_char,
                        end_char=clip.end_char
                    )
                    selected_clips.append(extended_clip)
                    print(f'Extended clip {i+1}: {extended_clip.start_time:.1f}s - {extended_clip.end_time:.1f}s (duration: {extended_clip.end_time - extended_clip.start_time:.1f}s)')
        else:
            # All clips are too long, trim the most engaging ones
            print('All clips are too long. Trimming most engaging clips to maximum duration...')
            long_clip_scores = [(clip, calculate_engagement_score(clip, transcription)) for clip in clips]
            long_clip_scores.sort(key=lambda x: x[1], reverse=True)
            # Take top 2 long clips and trim them
            for i, (clip, score) in enumerate(long_clip_scores[:2]):
                if clip.end_time - clip.start_time > MAX_CLIP_DURATION:
                    trimmed_clip = Clip(
                        start_time=clip.start_time,
                        end_time=clip.start_time + MAX_CLIP_DURATION,
                        start_char=clip.start_char,
                        end_char=clip.end_char
                    )
                    selected_clips.append(trimmed_clip)
                    print(f'Trimmed clip {i+1}: {trimmed_clip.start_time:.1f}s - {trimmed_clip.end_time:.1f}s (duration: {trimmed_clip.end_time - trimmed_clip.start_time:.1f}s)')

    # Process each selected clip
    for clip_index, clip in enumerate(selected_clips):
        print(f'\n--- Processing Clip {clip_index + 1}/{len(selected_clips)} ---')
        # 4. Trim the video to the selected clip
        trimmed_path = os.path.join(OUTPUT_DIR, f'trimmed_clip_{clip_index + 1}.mp4')
        print('Trimming video to selected clip...')
        if not trim_video_ffmpeg(
            input_path=input_path,
            output_path=trimmed_path,
            start_time=clip.start_time,
            end_time=clip.end_time,
        ):
            print("Failed to trim clip with ffmpeg. Skipping this clip.")
            continue
        # 5. Resize to 9:16 with strategy chain
        print(f"Resizing video to 9:16 (mode={RESIZE_MODE})...")
        resize_result = resize_with_strategy(
            mode=RESIZE_MODE,
            input_path=trimmed_path,
            output_dir=OUTPUT_DIR,
            clip_index=clip_index + 1,
            ai_resize_allowed=AI_RESIZE_ALLOWED,
            huggingface_token=HUGGINGFACE_TOKEN,
            aspect_ratio=(ASPECT_RATIO_WIDTH, ASPECT_RATIO_HEIGHT),
        )
        AI_RESIZE_ALLOWED = resize_result.ai_resize_allowed
        output_path = resize_result.output_path
        print(f"Resize strategy used: {resize_result.strategy_used} ({resize_result.details})")
        # 6. Add styled subtitles
        final_output = create_animated_subtitles(output_path, transcription, clip, output_path)
        # 7. Generate viral title using Groq API
        clip_text = " ".join([w["word"] for w in transcription.get_word_info() if w["start_time"] >= clip.start_time and w["end_time"] <= clip.end_time])
        groq_api_key = os.getenv('GROQ_API_KEY', 'your_groq_api_key_here')
        title = get_viral_title(clip_text, groq_api_key)
        print(f"\nViral Title for Clip {clip_index + 1}: {title}")
        # 8. Save the final video with the viral title (keep spaces, punctuation, and emojis)
        import shutil
        viral_filename = safe_filename(title).strip() + ".mp4"
        viral_path = os.path.join(OUTPUT_DIR, viral_filename)
        shutil.copy(final_output, viral_path)
        print(f"Final video saved as: {viral_path}\n")
    processed_videos += 1

print(f"\nProcessing complete. Success: {processed_videos}, Skipped: {skipped_videos}, Total: {len(video_transcription_map)}")
