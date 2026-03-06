import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import tkinter as tk
import time
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText


PROJECT_DIR = Path(__file__).resolve().parent
PYTHON_EXE = PROJECT_DIR / "env311" / "Scripts" / "python.exe"
MAIN_PY = PROJECT_DIR / "main.py"
# Disabled by default to avoid killing long CPU-only stages.
# Set env NO_OUTPUT_TIMEOUT_SEC > 0 only if you explicitly want watchdog kills.
NO_OUTPUT_TIMEOUT_SEC = int(os.getenv("NO_OUTPUT_TIMEOUT_SEC", "0"))
SUPPORTED_TRANSCRIPTION_MODELS = ["medium", "large-v1", "large-v2"]
SUPPORTED_FORCED_LANGUAGES = ["en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt"]
SUPPORTED_TRANSCRIPTION_BACKENDS = ["clipsai", "faster_whisper"]
SUPPORTED_SUBTITLE_STYLE_MODES = ["auto", "manual"]
SUPPORTED_SUBTITLE_STYLE_PRESETS = ["bold_clean", "dramatic", "minimal", "newsflash"]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ClippedAI Launcher")
        self.geometry("980x640")
        self.resizable(True, True)

        self.input_dir = tk.StringVar(value=str(PROJECT_DIR / "input"))
        self.output_dir = tk.StringVar(value=str(PROJECT_DIR / "output"))
        self.resize_mode = tk.StringVar(value="local_ai")
        self.max_clips = tk.IntVar(value=1)
        self.model_name = tk.StringVar(value="medium")
        self.transcription_backend = tk.StringVar(value="faster_whisper")
        self.profile_name = tk.StringVar(value="Balanced")
        self.language_mode = tk.StringVar(value="auto")
        self.forced_language = tk.StringVar(value="")
        self.llm_correction = tk.BooleanVar(value=False)
        self.subtitle_style_mode = tk.StringVar(value="auto")
        self.subtitle_style_preset = tk.StringVar(value="bold_clean")

        self.video_vars = {}
        self.video_paths = {}
        self.video_status = {}
        self.video_progress = {}
        self.queue = queue.Queue()
        self.worker = None
        self.session_log = []

        self._build_ui()
        self._log(f"Launcher started. NO_OUTPUT_TIMEOUT_SEC={NO_OUTPUT_TIMEOUT_SEC}")
        self._refresh_videos()
        self.after(100, self._poll_queue)

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)

        ttk.Label(top, text="Input folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.input_dir, width=80).grid(row=0, column=1, padx=6, sticky="ew")
        ttk.Button(top, text="Browse", command=self._pick_input).grid(row=0, column=2)

        ttk.Label(top, text="Output folder").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.output_dir, width=80).grid(row=1, column=1, padx=6, sticky="ew")
        ttk.Button(top, text="Browse", command=self._pick_output).grid(row=1, column=2)

        opts = ttk.Frame(self)
        opts.pack(fill="x", padx=10, pady=8)
        ttk.Label(opts, text="Resize mode").grid(row=0, column=0, sticky="w")
        ttk.Combobox(opts, textvariable=self.resize_mode, values=["auto", "local_ai", "ffmpeg", "ai"], width=12, state="readonly").grid(row=0, column=1, padx=6)
        ttk.Label(opts, text="Max clips/video").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(opts, from_=1, to=12, textvariable=self.max_clips, width=6).grid(row=0, column=3, padx=6)
        ttk.Label(opts, text="Transcription model").grid(row=0, column=4, sticky="w")
        ttk.Combobox(opts, textvariable=self.model_name, values=SUPPORTED_TRANSCRIPTION_MODELS, width=12, state="readonly").grid(row=0, column=5, padx=6)
        ttk.Label(opts, text="Backend").grid(row=0, column=6, sticky="w")
        ttk.Combobox(opts, textvariable=self.transcription_backend, values=SUPPORTED_TRANSCRIPTION_BACKENDS, width=14, state="readonly").grid(row=0, column=7, padx=6)
        ttk.Button(opts, text="Scan videos", command=self._refresh_videos).grid(row=0, column=8, padx=8)
        ttk.Label(opts, text="Language").grid(row=1, column=3, sticky="w", pady=(8, 0))
        ttk.Combobox(opts, textvariable=self.language_mode, values=["auto", "forced"], width=10, state="readonly").grid(row=1, column=4, padx=6, pady=(8, 0), sticky="w")
        ttk.Entry(opts, textvariable=self.forced_language, width=8).grid(row=1, column=5, padx=6, pady=(8, 0), sticky="w")
        ttk.Label(opts, text="Subtitle style").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(opts, textvariable=self.subtitle_style_mode, values=SUPPORTED_SUBTITLE_STYLE_MODES, width=10, state="readonly").grid(row=2, column=1, padx=6, pady=(8, 0), sticky="w")
        ttk.Combobox(opts, textvariable=self.subtitle_style_preset, values=SUPPORTED_SUBTITLE_STYLE_PRESETS, width=14, state="readonly").grid(row=2, column=2, padx=6, pady=(8, 0), sticky="w")
        ttk.Label(opts, text="Profile").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            opts,
            textvariable=self.profile_name,
            values=["Fast CPU", "Balanced", "Quality CPU"],
            width=14,
            state="readonly",
        ).grid(row=1, column=1, sticky="w", pady=(8, 0))
        ttk.Button(opts, text="Apply profile", command=self._apply_profile).grid(row=1, column=2, padx=6, pady=(8, 0), sticky="w")
        ttk.Checkbutton(opts, text="LLM text correction", variable=self.llm_correction).grid(row=1, column=6, padx=8, pady=(8, 0), sticky="w")

        middle = ttk.LabelFrame(self, text="Videos")
        middle.pack(fill="both", expand=True, padx=10, pady=6)

        self.tree = ttk.Treeview(middle, columns=("selected", "status", "progress"), show="headings", height=16)
        self.tree.heading("selected", text="Selected")
        self.tree.heading("status", text="Status")
        self.tree.heading("progress", text="Progress")
        self.tree.column("selected", width=80, anchor="center")
        self.tree.column("status", width=620, anchor="w")
        self.tree.column("progress", width=120, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=8, pady=8)

        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=10, pady=8)
        ttk.Button(bottom, text="Check env", command=self._check_environment).pack(side="left", padx=4)
        ttk.Button(bottom, text="Select all", command=self._select_all).pack(side="left", padx=4)
        ttk.Button(bottom, text="Clear", command=self._clear_all).pack(side="left", padx=4)
        ttk.Button(bottom, text="Copy log", command=self._copy_log).pack(side="left", padx=4)
        ttk.Button(bottom, text="Clear log", command=self._clear_log).pack(side="left", padx=4)
        ttk.Button(bottom, text="Start", command=self._start).pack(side="right", padx=4)

        self.overall = ttk.Progressbar(bottom, orient="horizontal", mode="determinate")
        self.overall.pack(fill="x", padx=8, expand=True, side="right")
        self.overall_label = ttk.Label(bottom, text="Overall: 0%")
        self.overall_label.pack(side="right", padx=8)

        log_frame = ttk.LabelFrame(self, text="Service Log")
        log_frame.pack(fill="both", expand=True, padx=10, pady=8)
        self.log_text = ScrolledText(log_frame, height=12, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=6, pady=6)
        self.log_text.configure(state="disabled")

    def _pick_input(self):
        p = filedialog.askdirectory(initialdir=self.input_dir.get())
        if p:
            self.input_dir.set(p)
            self._refresh_videos()

    def _pick_output(self):
        p = filedialog.askdirectory(initialdir=self.output_dir.get())
        if p:
            self.output_dir.set(p)

    def _refresh_videos(self):
        self._log(f"Scanning input folder: {self.input_dir.get()}")
        self.tree.delete(*self.tree.get_children())
        self.video_vars.clear()
        self.video_paths.clear()
        self.video_status.clear()
        self.video_progress.clear()
        input_path = Path(self.input_dir.get())
        if not input_path.exists():
            return
        files = sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"])
        self._log(f"Found {len(files)} mp4 file(s).")
        for i, f in enumerate(files, start=1):
            key = f"v{i}"
            self.video_vars[key] = tk.BooleanVar(value=True)
            self.video_paths[key] = f
            self.video_status[key] = f"Pending - {f.name}"
            self.video_progress[key] = 0
            self.tree.insert("", "end", iid=key, values=("Yes", self.video_status[key], "0%"))
        self.tree.bind("<Double-1>", self._toggle_selected)

    def _toggle_selected(self, event):
        item = self.tree.identify_row(event.y)
        if not item:
            return
        if item not in self.video_vars:
            return
        var = self.video_vars[item]
        var.set(not var.get())
        self._update_row(item)

    def _update_row(self, name):
        selected = "Yes" if self.video_vars[name].get() else "No"
        self.tree.item(name, values=(selected, self.video_status.get(name, ""), f"{self.video_progress.get(name, 0)}%"))

    def _select_all(self):
        self._log("Select all clicked.")
        for k in self.tree.get_children():
            self.video_vars[k].set(True)
            self._update_row(k)

    def _clear_all(self):
        self._log("Clear selection clicked.")
        for k in self.tree.get_children():
            self.video_vars[k].set(False)
            self._update_row(k)

    def _copy_log(self):
        data = self.log_text.get("1.0", "end-1c")
        self.clipboard_clear()
        self.clipboard_append(data)
        self._log("Log copied to clipboard.")

    def _clear_log(self):
        self.session_log.clear()
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _read_dotenv_value(self, key: str) -> str:
        env_file = PROJECT_DIR / ".env"
        if not env_file.exists():
            return ""
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                if k.strip() == key:
                    return v.strip().strip('"').strip("'")
        except Exception:
            return ""
        return ""

    def _resolve_tool_path(self, env_key: str, tool_name: str) -> str:
        explicit = os.getenv(env_key, "").strip() or self._read_dotenv_value(env_key)
        if explicit:
            return explicit
        found = shutil.which(tool_name)
        return found or ""

    def _validate_ffmpeg_tools(self):
        ffmpeg = self._resolve_tool_path("FFMPEG_EXE", "ffmpeg")
        ffprobe = self._resolve_tool_path("FFPROBE_EXE", "ffprobe")
        if not ffmpeg or not ffprobe:
            return False, (
                "FFmpeg/FFprobe not found.\n\n"
                "Set FFMPEG_EXE and FFPROBE_EXE in .env or add ffmpeg to PATH."
            )
        if not Path(ffmpeg).exists() or not Path(ffprobe).exists():
            return False, (
                "Configured ffmpeg path is invalid.\n\n"
                f"FFMPEG_EXE={ffmpeg}\nFFPROBE_EXE={ffprobe}"
            )
        return True, f"ffmpeg={ffmpeg}; ffprobe={ffprobe}"

    def _validate_runtime_environment(self):
        checks = []
        ok = True

        env_file = PROJECT_DIR / ".env"
        checks.append((env_file.exists(), f".env present: {env_file}"))
        if not env_file.exists():
            ok = False

        ff_ok, ff_msg = self._validate_ffmpeg_tools()
        checks.append((ff_ok, f"ffmpeg tools: {ff_msg}"))
        if not ff_ok:
            ok = False

        input_path = Path(self.input_dir.get())
        input_ok = input_path.exists() and input_path.is_dir()
        checks.append((input_ok, f"input dir: {input_path}"))
        if not input_ok:
            ok = False

        output_path = Path(self.output_dir.get())
        output_ok = True
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            probe_file = output_path / "write_probe.txt"
            probe_file.write_text("ok", encoding="utf-8")
            probe_file.unlink(missing_ok=True)
        except Exception as e:
            output_ok = False
            checks.append((False, f"output dir write failed: {output_path} ({e})"))
            ok = False
        if output_ok:
            checks.append((True, f"output dir writable: {output_path}"))

        hf_token = os.getenv("HUGGINGFACE_TOKEN", "").strip() or self._read_dotenv_value("HUGGINGFACE_TOKEN")
        groq_token = os.getenv("GROQ_API_KEY", "").strip() or self._read_dotenv_value("GROQ_API_KEY")
        hf_ok = bool(hf_token and hf_token != "your_huggingface_token_here")
        groq_ok = bool(groq_token and groq_token != "your_groq_api_key_here")
        checks.append((hf_ok, "HUGGINGFACE_TOKEN configured"))
        checks.append((groq_ok, "GROQ_API_KEY configured"))
        if not hf_ok:
            ok = False
        if self.llm_correction.get() and not groq_ok:
            checks.append((False, "LLM text correction requires GROQ_API_KEY"))
            ok = False

        lang_mode = self.language_mode.get().strip().lower()
        forced_lang = self.forced_language.get().strip().lower()
        lang_ok = lang_mode in {"auto", "forced"}
        if lang_mode == "forced":
            lang_ok = forced_lang in SUPPORTED_FORCED_LANGUAGES
        checks.append((lang_ok, f"language mode: {lang_mode}, forced={forced_lang or '-'}"))
        if not lang_ok:
            ok = False

        model_ok = self.model_name.get() in SUPPORTED_TRANSCRIPTION_MODELS
        checks.append((model_ok, f"transcription model: {self.model_name.get()}"))
        if not model_ok:
            ok = False

        backend_ok = self.transcription_backend.get() in SUPPORTED_TRANSCRIPTION_BACKENDS
        checks.append((backend_ok, f"transcription backend: {self.transcription_backend.get()}"))
        if not backend_ok:
            ok = False

        style_mode_ok = self.subtitle_style_mode.get() in SUPPORTED_SUBTITLE_STYLE_MODES
        style_preset_ok = self.subtitle_style_preset.get() in SUPPORTED_SUBTITLE_STYLE_PRESETS
        checks.append((style_mode_ok, f"subtitle style mode: {self.subtitle_style_mode.get()}"))
        checks.append((style_preset_ok, f"subtitle style preset: {self.subtitle_style_preset.get()}"))
        if not style_mode_ok or not style_preset_ok:
            ok = False

        return ok, checks

    def _check_environment(self):
        ok, checks = self._validate_runtime_environment()
        lines = []
        for item_ok, msg in checks:
            prefix = "OK" if item_ok else "FAIL"
            lines.append(f"[{prefix}] {msg}")
            self._log(f"Env check {prefix}: {msg}")
        text = "\n".join(lines)
        if ok:
            messagebox.showinfo("Environment check", text)
        else:
            messagebox.showwarning("Environment check", text)

    def _log(self, message: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        self.session_log.append(line)
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _start(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "Processing is already running.")
            self._log("Start rejected: process already running.")
            return
        selected_keys = [k for k in self.tree.get_children() if self.video_vars.get(k) and self.video_vars[k].get()]
        if not selected_keys:
            messagebox.showwarning("No videos", "Select at least one video.")
            self._log("Start rejected: no videos selected.")
            return
        env_ok, env_checks = self._validate_runtime_environment()
        if not env_ok:
            self._log("Start rejected: environment checks failed. Use 'Check env' for details.")
            messagebox.showerror("Environment not ready", "Environment checks failed. Click 'Check env' to view details.")
            return
        tools_ok, tools_msg = self._validate_ffmpeg_tools()
        if not tools_ok:
            messagebox.showerror("FFmpeg not configured", tools_msg)
            self._log(f"Start rejected: {tools_msg.replace(chr(10), ' ')}")
            return
        self._log(
            f"Start processing: {len(selected_keys)} video(s), mode={self.resize_mode.get()}, "
            f"backend={self.transcription_backend.get()}, max_clips={int(self.max_clips.get())}, model={self.model_name.get()}, "
            f"language_mode={self.language_mode.get()}, forced_language={self.forced_language.get().strip().lower() or '-'}, "
            f"subtitle_style={self.subtitle_style_mode.get()}/{self.subtitle_style_preset.get()}, "
            f"llm_correction={self.llm_correction.get()}, "
            f"timeout={'disabled' if NO_OUTPUT_TIMEOUT_SEC <= 0 else str(NO_OUTPUT_TIMEOUT_SEC)+'s'}."
        )
        if self.transcription_backend.get() == "clipsai" and self.model_name.get() == "large-v2":
            warning = "Warning: large-v2 with clipsai on CPU may take very long without visible progress."
            self._log(warning)
            messagebox.showwarning("Long CPU run expected", warning)
        self._log(f"Using tools: {tools_msg}")
        out_dir = Path(self.output_dir.get())
        out_dir.mkdir(parents=True, exist_ok=True)
        self.worker = threading.Thread(target=self._run_jobs, args=(selected_keys,), daemon=True)
        self.worker.start()

    def _apply_profile(self):
        p = self.profile_name.get()
        if p == "Fast CPU":
            self.model_name.set("medium")
            self.resize_mode.set("ffmpeg")
            self.max_clips.set(1)
            self.transcription_backend.set("faster_whisper")
            self.language_mode.set("auto")
            self.llm_correction.set(False)
            self.subtitle_style_mode.set("manual")
            self.subtitle_style_preset.set("bold_clean")
        elif p == "Quality CPU":
            self.model_name.set("large-v2")
            self.resize_mode.set("local_ai")
            self.max_clips.set(2)
            self.transcription_backend.set("faster_whisper")
            self.language_mode.set("auto")
            self.llm_correction.set(False)
            self.subtitle_style_mode.set("auto")
            self.subtitle_style_preset.set("bold_clean")
        else:
            self.model_name.set("medium")
            self.resize_mode.set("local_ai")
            self.max_clips.set(1)
            self.transcription_backend.set("faster_whisper")
            self.language_mode.set("auto")
            self.llm_correction.set(False)
            self.subtitle_style_mode.set("auto")
            self.subtitle_style_preset.set("bold_clean")
        self._log(
            f"Profile applied: {p} (backend={self.transcription_backend.get()}, model={self.model_name.get()}, "
            f"resize={self.resize_mode.get()}, max_clips={int(self.max_clips.get())}, "
            f"language_mode={self.language_mode.get()}, subtitle_style={self.subtitle_style_mode.get()}/{self.subtitle_style_preset.get()}, "
            f"llm_correction={self.llm_correction.get()})"
        )

    def _run_jobs(self, video_keys):
        total = len(video_keys)
        for idx, key in enumerate(video_keys, start=1):
            src_video = self.video_paths[key]
            video_name = src_video.name
            self.queue.put(("status", key, f"Preparing - {video_name}", 5))
            self.queue.put(("log", f"{video_name}: preparing temp workspace"))
            with tempfile.TemporaryDirectory(prefix="clippedai_job_") as td:
                td_path = Path(td)
                in_dir = td_path / "input"
                out_dir = td_path / "output"
                in_dir.mkdir(parents=True, exist_ok=True)
                out_dir.mkdir(parents=True, exist_ok=True)

                shutil.copy2(src_video, in_dir / video_name)
                pkl = src_video.with_name(src_video.stem + "_transcription.pkl")
                if pkl.exists():
                    shutil.copy2(pkl, in_dir / pkl.name)
                    self.queue.put(("log", f"{video_name}: existing transcription copied"))

                env = os.environ.copy()
                env["INPUT_DIR"] = str(in_dir)
                env["OUTPUT_DIR"] = str(out_dir)
                env["RESIZE_MODE"] = self.resize_mode.get()
                env["MAX_CLIPS_OVERRIDE"] = str(int(self.max_clips.get()))
                env["TRANSCRIPTION_BACKEND"] = self.transcription_backend.get()
                env["TRANSCRIPTION_MODEL"] = self.model_name.get()
                env["TRANSCRIPTION_LANGUAGE_MODE"] = self.language_mode.get().strip().lower()
                env["FORCED_LANGUAGE"] = self.forced_language.get().strip().lower()
                env["LLM_TEXT_CORRECTION"] = "true" if self.llm_correction.get() else "false"
                env["SUBTITLE_STYLE_MODE"] = self.subtitle_style_mode.get().strip().lower()
                env["SUBTITLE_STYLE_PRESET"] = self.subtitle_style_preset.get().strip().lower()

                cmd = [str(PYTHON_EXE), "-u", str(MAIN_PY)]
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_DIR),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )

                progress = 10
                self.queue.put(("status", key, f"Running - {video_name}", progress))
                self.queue.put(("log", f"{video_name}: started subprocess"))
                lines = []
                last_output = time.time()
                last_real_output = time.time()
                line_q = queue.Queue()

                def _reader():
                    try:
                        for raw in proc.stdout:
                            line_q.put(raw.rstrip("\r\n"))
                    finally:
                        line_q.put(None)

                reader_t = threading.Thread(target=_reader, daemon=True)
                reader_t.start()

                stream_closed = False
                while True:
                    try:
                        item = line_q.get(timeout=0.2)
                        if item is None:
                            stream_closed = True
                        else:
                            line = item.strip()
                            self.queue.put(("log", f"{video_name}: {line}"))
                            last_output = time.time()
                            last_real_output = time.time()
                            lines.append(line)
                            if len(lines) > 300:
                                lines.pop(0)
                            if "Found existing transcription" in line or "Transcribing video" in line:
                                progress = max(progress, 25)
                                self.queue.put(("status", key, f"Transcription - {video_name}", progress))
                            elif "Selected top" in line:
                                progress = max(progress, 45)
                                self.queue.put(("status", key, f"Clip selection - {video_name}", progress))
                            elif "Trimming video to selected clip" in line:
                                progress = max(progress, 60)
                                self.queue.put(("status", key, f"Trimming - {video_name}", progress))
                            elif "Resizing video to 9:16" in line:
                                progress = max(progress, 75)
                                self.queue.put(("status", key, f"Resizing - {video_name}", progress))
                            elif "Creating styled subtitles" in line:
                                progress = max(progress, 88)
                                self.queue.put(("status", key, f"Subtitles - {video_name}", progress))
                            elif "Final video saved as:" in line:
                                progress = 100
                                self.queue.put(("status", key, f"Completed - {video_name}", progress))
                    except queue.Empty:
                        pass

                    rc_now = proc.poll()
                    if rc_now is not None and stream_closed:
                        break

                    if time.time() - last_output > 20:
                        progress = min(95, progress + 1)
                        self.queue.put(("status", key, f"Running - {video_name}", progress))
                        last_output = time.time()
                        self.queue.put(("log", f"{video_name}: heartbeat progress {progress}%"))

                    if NO_OUTPUT_TIMEOUT_SEC > 0 and (time.time() - last_real_output > NO_OUTPUT_TIMEOUT_SEC):
                        proc.kill()
                        lines.append(f"Timeout: no output for {NO_OUTPUT_TIMEOUT_SEC} seconds.")
                        self.queue.put(
                            ("log", f"{video_name}: timeout (no output for {NO_OUTPUT_TIMEOUT_SEC}s), process killed")
                        )
                        break

                rc = proc.wait()
                if rc == 0:
                    self.queue.put(("log", f"{video_name}: subprocess finished successfully"))
                    for f in out_dir.glob("*.mp4"):
                        shutil.copy2(f, Path(self.output_dir.get()) / f.name)
                        self.queue.put(("log", f"{video_name}: copied output {f.name}"))
                else:
                    log_path = Path(self.output_dir.get()) / f"{src_video.stem}_error.log"
                    try:
                        log_path.write_text("\n".join(lines), encoding="utf-8")
                    except Exception:
                        pass
                    self.queue.put(("log", f"{video_name}: failed with exit {rc}, error log: {log_path}"))
                    self.queue.put(("status", key, f"Failed (exit={rc}) - see {log_path.name}", progress))

            overall = int((idx / total) * 100)
            self.queue.put(("overall", overall))

        self.queue.put(("done",))

    def _poll_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                kind = msg[0]
                if kind == "status":
                    _, name, status, p = msg
                    self.video_status[name] = status
                    self.video_progress[name] = p
                    self._update_row(name)
                elif kind == "overall":
                    overall = msg[1]
                    self.overall["value"] = overall
                    self.overall_label.config(text=f"Overall: {overall}%")
                elif kind == "log":
                    self._log(msg[1])
                elif kind == "done":
                    # Save full session log for easy sharing.
                    try:
                        out = Path(self.output_dir.get())
                        out.mkdir(parents=True, exist_ok=True)
                        p = out / f"launcher_session_{time.strftime('%Y%m%d_%H%M%S')}.log"
                        p.write_text("\n".join(self.session_log), encoding="utf-8")
                        self._log(f"Session log saved: {p}")
                    except Exception:
                        pass
                    messagebox.showinfo("Done", "Processing finished.")
        except queue.Empty:
            pass
        self.after(120, self._poll_queue)


if __name__ == "__main__":
    App().mainloop()
