"""Resize strategy module for robust 9:16 conversion."""

from __future__ import annotations

import glob
import os
import subprocess
import tempfile
from dataclasses import dataclass
from statistics import median
from typing import List, Optional, Tuple

import cv2


@dataclass
class ResizeResult:
    output_path: str
    strategy_used: str
    ai_resize_allowed: bool
    details: str = ""


def configure_ffmpeg() -> Tuple[str, str]:
    """Find ffmpeg/ffprobe and ensure they are available in PATH."""
    def _can_run(exe: str) -> bool:
        try:
            subprocess.run([exe, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            return True
        except Exception:
            return False

    # Explicit overrides for environments where PATH is unreliable.
    env_ffmpeg = os.getenv("FFMPEG_EXE")
    env_ffprobe = os.getenv("FFPROBE_EXE")
    if env_ffmpeg and env_ffprobe and _can_run(env_ffmpeg) and _can_run(env_ffprobe):
        return env_ffmpeg, env_ffprobe

    ffmpeg_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    ffprobe_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    if _can_run(ffmpeg_name) and _can_run(ffprobe_name):
        return ffmpeg_name, ffprobe_name

    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    for d in path_dirs:
        if not d:
            continue
        ffmpeg_path = os.path.join(d, ffmpeg_name)
        ffprobe_path = os.path.join(d, ffprobe_name)
        if os.path.isfile(ffmpeg_path) and os.path.isfile(ffprobe_path):
            if not (_can_run(ffmpeg_path) and _can_run(ffprobe_path)):
                continue
            return ffmpeg_path, ffprobe_path

    candidates = []
    if os.name == "nt":
        candidates.extend([
            r"C:\ffmpeg\bin",
            r"C:\Program Files\ffmpeg\bin",
            os.path.expandvars(
                r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-*\bin"
            ),
        ])

    for candidate in candidates:
        for d in glob.glob(candidate):
            ffmpeg_path = os.path.join(d, ffmpeg_name)
            ffprobe_path = os.path.join(d, ffprobe_name)
            if os.path.isfile(ffmpeg_path) and os.path.isfile(ffprobe_path):
                if not (_can_run(ffmpeg_path) and _can_run(ffprobe_path)):
                    continue
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                return ffmpeg_path, ffprobe_path

    return "", ""


def _run_ffmpeg(ffmpeg_exe: str, input_path: str, output_path: str, vf: str) -> bool:
    cmd = [ffmpeg_exe, "-y", "-i", input_path, "-vf", vf, "-c:a", "copy", output_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def resize_ffmpeg_center(ffmpeg_exe: str, input_path: str, output_path: str) -> bool:
    vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
    return _run_ffmpeg(ffmpeg_exe, input_path, output_path, vf)


def _estimate_subject_center_x(input_path: str) -> Optional[float]:
    """Estimate horizontal subject center by sampling faces with OpenCV."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        cap.release()
        return None

    centers = []
    frame_idx = 0
    sample_step = 15

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % sample_step != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )
            if len(faces) == 0:
                continue

            # Use largest face as primary subject.
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            centers.append(x + w / 2.0)
    finally:
        cap.release()

    if not centers:
        return None
    return float(median(centers))


def _dynamic_local_ai_render(
    ffmpeg_exe: str, input_path: str, output_path: str
) -> Tuple[bool, str]:
    """Dynamic subject-aware crop render using OpenCV face tracking + ffmpeg audio mux."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False, "failed to open video"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        return False, "invalid video dimensions"

    # For wide videos we crop width to 9:16 window; for vertical/square keep center behavior.
    crop_w = min(width, int(height * 9 / 16))
    crop_h = min(height, int(width * 16 / 9)) if width < int(height * 9 / 16) else height

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        cap.release()
        return False, "face detector unavailable"

    fd, temp_video = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_video, fourcc, fps, (1080, 1920))
    if not writer.isOpened():
        cap.release()
        return False, "failed to create temp writer"

    sample_step = max(5, int(fps // 2))
    frame_idx = 0
    detected_frames = 0
    smooth_x: Optional[float] = None
    target_x: Optional[float] = None
    alpha = 0.25

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            if frame_idx % sample_step == 0 or target_x is None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
                )
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                    target_x = x + w / 2.0
                    detected_frames += 1

            if target_x is None:
                target_x = width / 2.0

            if smooth_x is None:
                smooth_x = target_x
            else:
                smooth_x = (1 - alpha) * smooth_x + alpha * target_x

            if width >= int(height * 9 / 16):
                x0 = int(max(0, min(width - crop_w, smooth_x - crop_w / 2)))
                y0 = 0
                crop = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]
            else:
                x0 = 0
                y0 = int(max(0, min(height - crop_h, (height - crop_h) / 2)))
                crop = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]

            if crop.size == 0:
                crop = frame

            out_frame = cv2.resize(crop, (1080, 1920), interpolation=cv2.INTER_LINEAR)
            writer.write(out_frame)
    finally:
        cap.release()
        writer.release()

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        temp_video,
        "-i",
        input_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-shortest",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        details = f"dynamic-track faces={detected_frames}"
        return True, details
    except Exception:
        return False, "audio mux failed"
    finally:
        try:
            os.remove(temp_video)
        except OSError:
            pass


def resize_local_ai(ffmpeg_exe: str, input_path: str, output_path: str) -> Tuple[bool, str]:
    """Local AI-like strategy: detect subject and crop toward it, then scale to 1080x1920."""
    dynamic_ok, dynamic_details = _dynamic_local_ai_render(ffmpeg_exe, input_path, output_path)
    if dynamic_ok:
        return True, dynamic_details

    subject_x = _estimate_subject_center_x(input_path)
    if subject_x is None:
        ok = resize_ffmpeg_center(ffmpeg_exe, input_path, output_path)
        return ok, "no face detected; fallback to ffmpeg center crop"

    # For wide videos, crop with subject-aware X. For tall videos, this expression keeps center behavior.
    # crop_w = if(gte(iw/ih,9/16), ih*9/16, iw)
    # crop_h = if(gte(iw/ih,9/16), ih, iw*16/9)
    # x for wide: clip(subject_x - crop_w/2, 0, iw-crop_w)
    vf = (
        "crop="
        "'if(gte(iw/ih,9/16),ih*9/16,iw)':"
        "'if(gte(iw/ih,9/16),ih,iw*16/9)':"
        f"'if(gte(iw/ih,9/16),clip({subject_x}-ih*9/32,0,iw-ih*9/16),0)':"
        "'if(gte(iw/ih,9/16),0,clip((ih-iw*16/9)/2,0,ih-iw*16/9))',"
        "scale=1080:1920"
    )
    ok = _run_ffmpeg(ffmpeg_exe, input_path, output_path, vf)
    details = f"static-face subject_x={subject_x:.1f}"
    return ok, details


def resize_with_strategy(
    mode: str,
    input_path: str,
    output_dir: str,
    clip_index: int,
    ai_resize_allowed: bool,
    huggingface_token: str,
    aspect_ratio: Tuple[int, int],
):
    """Run configured resize strategy with safe fallbacks."""
    ffmpeg_exe, _ffprobe_exe = configure_ffmpeg()
    mode = (mode or "auto").lower()

    ai_output = os.path.join(output_dir, f"yt_short_ai_{clip_index}.mp4")
    local_output = os.path.join(output_dir, f"yt_short_local_ai_{clip_index}.mp4")
    ffmpeg_output = os.path.join(output_dir, f"yt_short_ffmpeg_{clip_index}.mp4")

    def try_ai() -> Optional[ResizeResult]:
        nonlocal ai_resize_allowed
        if not ai_resize_allowed:
            return None
        try:
            from clipsai import resize as clipsai_resize
            from clipsai import MediaEditor, AudioVideoFile

            crops = clipsai_resize(
                video_file_path=input_path,
                pyannote_auth_token=huggingface_token,
                aspect_ratio=aspect_ratio,
            )
            media_editor = MediaEditor()
            media_editor.resize_video(
                original_video_file=AudioVideoFile(input_path),
                resized_video_file_path=ai_output,
                width=crops.crop_width,
                height=crops.crop_height,
                segments=crops.to_dict()["segments"],
            )
            return ResizeResult(ai_output, "ai", ai_resize_allowed, "clipsai/pyannote")
        except Exception as e:
            err = str(e).lower()
            if (
                "gated" in err
                or "restricted" in err
                or "pyannote/speaker-diarization-3.1" in err
                or "could not download" in err
            ):
                ai_resize_allowed = False
            return None

    def try_local_ai() -> Optional[ResizeResult]:
        ok, details = resize_local_ai(ffmpeg_exe, input_path, local_output)
        if ok:
            return ResizeResult(local_output, "local_ai", ai_resize_allowed, details)
        return None

    def try_ffmpeg() -> ResizeResult:
        ok = resize_ffmpeg_center(ffmpeg_exe, input_path, ffmpeg_output)
        if ok:
            return ResizeResult(ffmpeg_output, "ffmpeg", ai_resize_allowed, "center crop")
        return ResizeResult(input_path, "none", ai_resize_allowed, "resize failed, keeping trimmed clip")

    if mode == "ai":
        return try_ai() or try_ffmpeg()
    if mode == "local_ai":
        return try_local_ai() or try_ffmpeg()
    if mode == "ffmpeg":
        return try_ffmpeg()

    # auto: ai -> local_ai -> ffmpeg
    return try_ai() or try_local_ai() or try_ffmpeg()
