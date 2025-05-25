import tkinter as tk
from tkinter import filedialog, messagebox
import os
import json
import threading
import torch
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence

# Import ttkbootstrap for modern theme
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

# Import WhisperX and Pyannote
import whisperx
from pyannote.audio import Pipeline as DiarizationPipeline
from pyannote.core import Segment

class WhisperXGUI:
    def __init__(self, master):
        self.master = master
        master.title("WhisperX GUI")
        master.geometry("900x900") # Slightly larger window
        master.resizable(False, False) # Make window not resizable

        self.diarization_pipeline = None # Will be loaded on demand
        self.whisper_model = None # Will be loaded on demand
        self._loaded_model_name = None # To track the currently loaded model name
        self._loaded_device = None # To track the currently loaded device
        self.current_compute_type = None # Initialize compute_type here
        self._transcription_cancelled = False # Flag to stop transcription

        self.settings = self.load_settings()
        # Set default output directory if not already set
        if 'output_directory' not in self.settings:
            self.settings['output_directory'] = os.path.join(os.getcwd(), "output")
        
        # Initialize i18n
        self.current_language = self.settings.get('language', 'en')
        self.translations = self.load_translations(self.current_language)

        # Main Frame
        self.main_frame = ttk.Frame(master, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # File Selection
        self.file_frame = ttk.LabelFrame(self.main_frame, text=self.translate("file_selection_frame_text"), padding="10")
        self.file_frame.pack(fill=tk.X, pady=10)

        self.file_listbox = tk.Listbox(self.file_frame, height=5, selectmode=tk.MULTIPLE)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = ttk.Scrollbar(self.file_frame, orient="vertical", command=self.file_listbox.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.config(yscrollcommand=self.scrollbar.set)

        self.add_files_button = ttk.Button(self.file_frame, text=self.translate("add_files_button"), command=self.add_files, bootstyle="primary")
        self.add_files_button.pack(side=tk.TOP, padx=5, pady=5)
        self.delete_selected_files_button = ttk.Button(self.file_frame, text=self.translate("delete_selected_files_button"), command=self.delete_selected_files, bootstyle="danger")
        self.delete_selected_files_button.pack(side=tk.TOP, padx=5, pady=5)

        # Options Frame
        self.options_frame = ttk.LabelFrame(self.main_frame, text=self.translate("options_frame_text"), padding="10")
        self.options_frame.pack(fill=tk.X, pady=10)

        self.diarization_var = tk.BooleanVar(value=self.settings.get('diarization_enabled', False))
        self.diarization_checkbox = ttk.Checkbutton(self.options_frame, text=self.translate("enable_diarization_checkbox"), variable=self.diarization_var)
        self.diarization_checkbox.pack(anchor=tk.W, pady=5)

        # Language and Model selection (will be populated from settings)
        self.language_label = ttk.Label(self.options_frame, text=self.translate("language_label"))
        self.language_label.pack(anchor=tk.W, pady=2)
        self.languages_map = {
            "自動偵測": "auto", "Afrikaans": "af", "Albanian": "sq", "Amharic": "am", "Arabic": "ar",
            "Armenian": "hy", "Assamese": "as", "Azerbaijani": "az", "Bashkir": "ba", "Basque": "eu",
            "Belarusian": "be", "Bengali": "bn", "Bosnian": "bs", "Breton": "br", "Bulgarian": "bg",
            "Burmese": "my", "Castilian": "es", "Catalan": "ca", "Chinese": "zh", "Croatian": "hr",
            "Czech": "cs", "Danish": "da", "Dutch": "nl", "English": "en", "Estonian": "et",
            "Faroese": "fo", "Finnish": "fi", "Flemish": "nl", "French": "fr", "Galician": "gl",
            "Georgian": "ka", "German": "de", "Greek": "el", "Gujarati": "gu", "Haitian": "ht",
            "Haitian Creole": "ht", "Hausa": "ha", "Hawaiian": "haw", "Hebrew": "he", "Hindi": "hi",
            "Hungarian": "hu", "Icelandic": "is", "Indonesian": "id", "Italian": "it", "Japanese": "ja",
            "Javanese": "jw", "Kannada": "kn", "Kazakh": "kk", "Khmer": "km", "Korean": "ko",
            "Lao": "lo", "Latin": "la", "Latvian": "lv", "Letzeburgesch": "lb", "Lingala": "ln",
            "Lithuanian": "lt", "Luxembourgish": "lb",
            "Macedonian": "mk", "Malagasy": "mg", "Malay": "ms", "Malayalam": "ml", "Maltese": "mt",
            "Maori": "mi", "Marathi": "mr", "Moldavian": "mo", "Moldovan": "mo", "Mongolian": "mn",
            "Myanmar": "my", "Nepali": "ne", "Norwegian": "no", "Nynorsk": "nn", "Occitan": "oc",
            "Panjabi": "pa", "Pashto": "ps", "Persian": "fa", "Polish": "pl", "Portuguese": "pt",
            "Punjabi": "pa", "Pushto": "ps", "Romanian": "ro", "Russian": "ru", "Sanskrit": "sa",
            "Serbian": "sr", "Shona": "sn", "Sindhi": "sd", "Sinhala": "si", "Sinhalese": "si",
            "Slovak": "sk", "Slovenian": "sl", "Somali": "so", "Spanish": "es", "Sundanese": "su",
            "Swahili": "sw", "Swedish": "sv", "Tagalog": "tl", "Tajik": "tg", "Tamil": "ta",
            "Tatar": "tt", "Telugu": "te", "Thai": "th", "Tibetan": "bo", "Turkish": "tr",
            "Turkmen": "tk", "Ukrainian": "uk", "Urdu": "ur", "Uzbek": "uz", "Valencian": "ca",
            "Vietnamese": "vi", "Welsh": "cy", "Yiddish": "yi", "Yoruba": "yo"
        }
        self.language_combobox = ttk.Combobox(self.options_frame, values=list(self.languages_map.keys()))
        # Set initial language based on saved settings, defaulting to "English" if not found
        initial_lang_code = self.settings.get('last_language', 'en')
        initial_lang_name = next((name for name, code in self.languages_map.items() if code == initial_lang_code), "English")
        self.language_combobox.set(initial_lang_name)
        self.language_combobox.pack(fill=tk.X, pady=2)

        self.model_label = ttk.Label(self.options_frame, text=self.translate("model_label"))
        self.model_label.pack(anchor=tk.W, pady=2)
        self.model_combobox = ttk.Combobox(self.options_frame, values=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"])
        self.model_combobox.set(self.settings.get('last_model', 'base'))
        self.model_combobox.pack(fill=tk.X, pady=2)

        # Initial Prompt input
        self.initial_prompt_label = ttk.Label(self.options_frame, text=self.translate("initial_prompt_label"))
        self.initial_prompt_label.pack(anchor=tk.W, pady=2)
        self.initial_prompt_entry = ttk.Entry(self.options_frame)
        self.initial_prompt_entry.insert(0, self.settings.get('initial_prompt', ''))
        self.initial_prompt_entry.pack(fill=tk.X, pady=2)

        # Device selection
        self.device_label = ttk.Label(self.options_frame, text=self.translate("device_label"))
        self.device_label.pack(anchor=tk.W, pady=2)
        self.available_devices = ["CPU"]
        if torch.cuda.is_available():
            self.available_devices.append("GPU")
        self.device_combobox = ttk.Combobox(self.options_frame, values=self.available_devices)
        
        # Set initial device based on saved settings, defaulting to "CPU" if not found or device not available
        initial_device_name = self.settings.get('last_device', 'CPU')
        if initial_device_name not in self.available_devices:
            initial_device_name = 'CPU' # Fallback to CPU if saved device is not available
        self.device_combobox.set(initial_device_name)
        self.device_combobox.pack(fill=tk.X, pady=2)

        # Action Buttons
        self.button_frame = ttk.Frame(self.main_frame, padding="10")
        self.button_frame.pack(fill=tk.X, pady=10)

        self.transcribe_button = ttk.Button(self.button_frame, text=self.translate("start_transcription_button"), command=self.start_transcription, bootstyle="success")
        self.transcribe_button.pack(side=tk.LEFT, expand=True, padx=5)

        self.cancel_button = ttk.Button(self.button_frame, text=self.translate("cancel_transcription_button"), command=self.cancel_transcription, bootstyle="warning", state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT, expand=True, padx=5)

        self.settings_button = ttk.Button(self.button_frame, text=self.translate("settings_button"), command=self.open_settings, bootstyle="info")
        self.settings_button.pack(side=tk.RIGHT, expand=True, padx=5)

        # Output Area
        self.output_frame = ttk.LabelFrame(self.main_frame, text=self.translate("output_area_frame_text"), padding="10")
        self.output_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Scrollbar first, packed to the right
        self.output_text_scrollbar = ttk.Scrollbar(self.output_frame, orient="vertical")
        self.output_text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Text widget next, filling the remaining space
        self.output_text = tk.Text(self.output_frame, wrap=tk.WORD, height=10, state=tk.DISABLED,
                                   yscrollcommand=self.output_text_scrollbar.set) # Link scrollbar here
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True) # Pack to left or just fill

        # Configure scrollbar's command after text widget is created
        self.output_text_scrollbar.config(command=self.output_text.yview)

        # Progress Bar
        self.progress_frame = ttk.LabelFrame(self.main_frame, text=self.translate("progress_frame_text"), padding="10")
        self.progress_frame.pack(fill=tk.X, pady=10)
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient="horizontal", length=100, mode="determinate")
        self.progress_bar.pack(fill=tk.X, expand=True, pady=5)
        self.progress_label = ttk.Label(self.progress_frame, text=self.translate("progress_label_ready"))
        self.progress_label.pack(pady=2)

    def load_settings(self):
        settings_path = "settings.json"
        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                messagebox.showerror(self.translate("settings_error_title"), self.translate("settings_error_message"))
                return {}
        return {}

    def save_settings(self):
        settings_path = "settings.json"
        with open(settings_path, 'w') as f:
            json.dump(self.settings, f, indent=4)

    def load_translations(self, lang_code):
        lang_file = os.path.join("lang", f"{lang_code}.json")
        if os.path.exists(lang_file):
            with open(lang_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {} # Fallback to empty if file not found

    def translate(self, key):
        return self.translations.get(key, key) # Return key itself if translation not found

    def add_files(self):
        files = filedialog.askopenfilenames(
            title="Select Audio/Video Files",
            filetypes=[("Audio/Video Files", "*.mp3 *.wav *.flac *.m4a *.mp4 *.avi *.mov"), ("All Files", "*.*")]
        )
        for file in files:
            if file not in self.file_listbox.get(0, tk.END):
                self.file_listbox.insert(tk.END, file)

    def delete_selected_files(self):
        selected_indices = self.file_listbox.curselection()
        for i in reversed(selected_indices): # Delete from end to beginning to avoid index issues
            self.file_listbox.delete(i)

    def start_transcription(self):
        selected_files = self.file_listbox.get(0, tk.END)
        if not selected_files:
            messagebox.showwarning("No Files Selected", "Please add audio/video files to transcribe.")
            return

        # Save current language, model, initial prompt, and device to settings
        self.settings['last_language'] = self.languages_map.get(self.language_combobox.get(), 'en') # Save code
        self.settings['last_model'] = self.model_combobox.get()
        self.settings['initial_prompt'] = self.initial_prompt_entry.get() # Save initial prompt
        self.settings['last_device'] = self.device_combobox.get() # Save the display name of the device
        self.settings['diarization_enabled'] = self.diarization_var.get()
        self.save_settings()

        self.output_text.config(state=tk.NORMAL) # Enable editing
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "Starting transcription...\n")
        self.output_text.insert(tk.END, f"Diarization Enabled: {self.diarization_var.get()}\n")
        self.output_text.insert(tk.END, f"Language: {self.language_combobox.get()}\n")
        self.output_text.insert(tk.END, f"Model: {self.model_combobox.get()}\n")
        self.output_text.insert(tk.END, f"Initial Prompt: {self.initial_prompt_entry.get()}\n") # Display initial prompt
        self.output_text.insert(tk.END, f"Device: {self.device_combobox.get()}\n\n")
        self.output_text.config(state=tk.DISABLED) # Disable editing

        self.transcribe_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL) # Enable cancel button
        self.settings_button.config(state=tk.DISABLED)
        self.add_files_button.config(state=tk.DISABLED)
        self.delete_selected_files_button.config(state=tk.DISABLED) # Update button name
        self._transcription_cancelled = False # Reset cancellation flag

        # Run transcription in a separate thread to keep GUI responsive
        threading.Thread(target=self._run_transcription_process, args=(selected_files,)).start()

    def cancel_transcription(self):
        self._transcription_cancelled = True
        self.update_progress(0, 1, self.translate("cancelling_progress"))
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, self.translate("transcription_cancellation_requested"))
        self.output_text.config(state=tk.DISABLED)
        self.master.update_idletasks()

    def _run_transcription_process(self, selected_files):
        try:
            total_files = len(selected_files)
            self.progress_bar["maximum"] = total_files
            self.progress_bar["value"] = 0

            for i, file_path in enumerate(selected_files):
                if self._transcription_cancelled:
                    self.output_text.config(state=tk.NORMAL)
                    self.output_text.insert(tk.END, self.translate("transcription_cancelled_by_user"))
                    self.output_text.config(state=tk.DISABLED)
                    break # Exit the loop if cancelled

                self.update_progress(i + 1, total_files, f"Processing: {os.path.basename(file_path)}")
                self.output_text.config(state=tk.NORMAL)
                self.output_text.insert(tk.END, f"Processing file {i+1}/{total_files}: {os.path.basename(file_path)}\n")
                self.output_text.config(state=tk.DISABLED)
                self.master.update_idletasks()

                try:
                    output_format = self.settings.get('output_format', 'txt')
                    save_to_input_dir = self.settings.get('save_to_input_dir', False)

                    transcribed_result = self.transcribe_audio(file_path)

                    base_filename = os.path.splitext(os.path.basename(file_path))[0]
                    if save_to_input_dir:
                        output_dir = os.path.dirname(file_path)
                    else:
                        # Default output directory to 'output' folder under program directory
                        default_output_dir = os.path.join(os.getcwd(), "output")
                        output_dir = self.settings.get('output_directory', default_output_dir)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)

                    selected_output_formats = self.settings.get('output_formats', ['txt']) # Get selected formats

                    for current_output_format in selected_output_formats:
                        output_filename = os.path.join(output_dir, f"{base_filename}.{current_output_format}")

                        # Handle different output formats
                        if current_output_format == 'txt':
                            # Extract text from segments for TXT output
                            text_content = "\n".join([s["text"] for s in transcribed_result])
                            with open(output_filename, "w", encoding="utf-8") as f:
                                f.write(text_content)
                        elif current_output_format == 'srt':
                            srt_content = self._format_to_srt(transcribed_result)
                            with open(output_filename, "w", encoding="utf-8") as f:
                                f.write(srt_content)
                        elif current_output_format == 'vtt':
                            vtt_content = self._format_to_vtt(transcribed_result)
                            with open(output_filename, "w", encoding="utf-8") as f:
                                f.write(vtt_content)

                        self.output_text.config(state=tk.NORMAL)
                        self.output_text.insert(tk.END, f"  Transcription for {os.path.basename(file_path)} saved to {output_filename}\n")
                        self.output_text.config(state=tk.DISABLED)
                    self.output_text.config(state=tk.NORMAL)
                    self.output_text.insert(tk.END, "\n") # Add an extra newline after all formats for a file
                    self.output_text.config(state=tk.DISABLED)

                except Exception as e:
                    self.output_text.config(state=tk.NORMAL)
                    self.output_text.insert(tk.END, f"  Error processing {os.path.basename(file_path)}: {e}\n\n")
                    self.output_text.config(state=tk.DISABLED)
                self.master.update_idletasks()

            if not self._transcription_cancelled:
                self.update_progress(total_files, total_files, "Transcription complete!")
                self.output_text.config(state=tk.NORMAL)
                self.output_text.insert(tk.END, "Transcription complete!\n")
                self.output_text.config(state=tk.DISABLED)
                messagebox.showinfo("Transcription Complete", "All selected files have been processed.")
            else:
                self.update_progress(0, 1, "Cancelled.")
                messagebox.showinfo("Transcription Cancelled", "Transcription process was cancelled.")


        except Exception as e:
            self.update_progress(0, 1, "Error occurred!")
            self.output_text.config(state=tk.NORMAL)
            self.output_text.insert(tk.END, f"\nAn unexpected error occurred during processing: {e}\n")
            self.output_text.config(state=tk.DISABLED)
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")
        finally:
            self.transcribe_button.config(state=tk.NORMAL)
            self.cancel_button.config(state=tk.DISABLED) # Disable cancel button
            self.settings_button.config(state=tk.NORMAL)
            self.add_files_button.config(state=tk.NORMAL)
            self.delete_selected_files_button.config(state=tk.NORMAL) # Update button name
            self.progress_label.config(text="Ready")

    def update_progress(self, value, maximum, text):
        self.progress_bar["value"] = value
        self.progress_bar["maximum"] = maximum
        self.progress_label.config(text=text)
        self.master.update_idletasks()

    def transcribe_audio(self, audio_path):
        model_name = self.model_combobox.get()
        language_display_name = self.language_combobox.get()
        language_code = self.languages_map.get(language_display_name, "en") # Get code from map
        initial_prompt = self.initial_prompt_entry.get() # Get initial prompt
        diarization_enabled = self.diarization_var.get()
        hf_token = self.settings.get('huggingface_token')
        selected_device_display_name = self.device_combobox.get()
        output_format = self.settings.get('output_format', 'txt')
        save_to_input_dir = self.settings.get('save_to_input_dir', False)

        # Determine the actual device string (e.g., "cuda:0" or "cpu") for whisperx and pyannote
        current_device = "cpu"
        self.current_compute_type = "int8" # Default to CPU and int8

        if selected_device_display_name == "GPU" and torch.cuda.is_available():
            current_device = "cuda"
            self.current_compute_type = "float16" # Use float16 for CUDA
        else: # Fallback to CPU if "GPU" not selected or CUDA not available
            current_device = "cpu"
            self.current_compute_type = "int8"

        # print(current_device)
        
        # Ensure models directory exists
        model_dir = "./models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.update_progress(0, 100, self.translate("loading_whisper_model"))
        # Check if model needs to be reloaded (different model name or device)
        if self.whisper_model is None or \
           self._loaded_model_name != model_name or \
           self._loaded_device != current_device:
            
            # Unload previous model if exists
            if self.whisper_model:
                del self.whisper_model
                if torch.cuda.is_available(): # Only clear cache if CUDA is available
                    torch.cuda.empty_cache()

            # Check for cancellation before loading model
            if self._transcription_cancelled:
                raise Exception(self.translate("transcription_cancelled_by_user_short"))

            self.whisper_model = whisperx.load_model(
                model_name,
                current_device, # Use current_device here
                compute_type=self.current_compute_type, # Use self.current_compute_type here
                language=language_code if language_code != "auto" else None,
                download_root=model_dir, # Specify model download directory
                asr_options={"initial_prompt": initial_prompt}
            )
            self._loaded_model_name = model_name # Store model name for caching
            self._loaded_device = current_device # Store device for caching

        audio = whisperx.load_audio(audio_path)

        self.update_progress(20, 100, self.translate("transcribing_audio"))
        
        transcribe_kwargs = {"batch_size": 8, "verbose": True}
        
        chunk_size = self.settings.get('chunk_size', 5) # Get chunk_size from settings
        transcribe_kwargs["chunk_size"] = chunk_size

        result = self.whisper_model.transcribe(audio, **transcribe_kwargs)

        # Conditionally align for better segment timing
        if self.settings.get('realign_audio', True): # Default to True if not set
            self.update_progress(40, 100, self.translate("aligning_transcription"))
            # 使用者選擇的語言，或者 whisper 模型轉錄後偵測到的語言
            align_language_code = result["language"]
            model_a, metadata = whisperx.load_align_model(language_code=align_language_code, device=current_device)

            # Check for cancellation before aligning
            if self._transcription_cancelled:
                raise Exception(self.translate("transcription_cancelled_by_user_short"))

            # 確保傳遞正確的參數給 align 函數
            # 參數順序: segments, alignment_model, alignment_model_metadata, audio, device
            result = whisperx.align(
                result["segments"], # list of transcribed segments
                model_a,            # alignment model
                metadata,           # alignment model metadata
                audio,              # audio waveform (numpy array from whisperx.load_audio)
                current_device,     # device string ("cuda:0" or "cpu")
                return_char_alignments=False,
            )

        if diarization_enabled:
            if not hf_token:
                raise ValueError(self.translate("hf_token_required_error"))
            if self.diarization_pipeline is None:
                self.update_progress(60, 100, self.translate("loading_diarization_model"))
                
                # Check for cancellation before loading diarization model
                if self._transcription_cancelled:
                    raise Exception(self.translate("transcription_cancelled_by_user_short"))
                
                self.diarization_pipeline = DiarizationPipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token, cache_dir="./models")
                self.diarization_pipeline.to(torch.device(current_device)) # Use current_device here

            self.update_progress(80, 100, self.translate("performing_speaker_diarization"))
            
            # Check for cancellation before diarization
            if self._transcription_cancelled:
                raise Exception(self.translate("transcription_cancelled_by_user_short"))

            diarize_segments = self.diarization_pipeline(audio_path)

            # # Convert pyannote Annotation to a list of dictionaries for whisperx
            # diarize_result = []
            # for segment, track, speaker in diarize_segments.itertracks(yield_label=True):
            #     diarize_result.append({"segment": Segment(segment.start, segment.end), "label": speaker})

            result = whisperx.assign_word_speakers(diarize_segments, result)

            # Format output with speakers
            formatted_segments = []
            current_speaker = None
            current_text = ""
            current_start = None
            current_end = None

            for segment in result["segments"]:
                speaker = segment.get("speaker", "UNKNOWN")
                text = segment["text"].strip()
                start = segment["start"]
                end = segment["end"]

                if speaker != current_speaker:
                    if current_text:
                        formatted_segments.append({
                            "speaker": current_speaker,
                            "text": current_text.strip(),
                            "start": current_start,
                            "end": current_end
                        })
                    current_speaker = speaker
                    current_text = text
                    current_start = start
                    current_end = end
                else:
                    current_text += " " + text
                    current_end = end # Extend the end time

            if current_text:
                formatted_segments.append({
                    "speaker": current_speaker,
                    "text": current_text.strip(),
                    "start": current_start,
                    "end": current_end
                })
            
            # Return the structured segments for formatting
            return formatted_segments
        else:
            # Return structured segments without speaker info
            return [{"text": segment["text"], "start": segment["start"], "end": segment["end"]} for segment in result["segments"]]

    def _format_to_srt(self, segments):
        srt_content = []
        for i, segment in enumerate(segments):
            start_time = self._format_timestamp(segment["start"])
            end_time = self._format_timestamp(segment["end"])
            text = segment.get("speaker", "") + ": " + segment["text"] if "speaker" in segment else segment["text"]
            srt_content.append(f"{i + 1}\n{start_time} --> {end_time}\n{text}\n")
        return "\n".join(srt_content)

    def _format_to_vtt(self, segments):
        vtt_content = ["WEBVTT\n"]
        for segment in segments:
            start_time = self._format_timestamp(segment["start"], vtt=True)
            end_time = self._format_timestamp(segment["end"], vtt=True)
            text = segment.get("speaker", "") + ": " + segment["text"] if "speaker" in segment else segment["text"]
            vtt_content.append(f"{start_time} --> {end_time}\n{text}\n")
        return "\n".join(vtt_content)

    def _format_timestamp(self, seconds, vtt=False):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds_int = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)

        if vtt:
            return f"{hours:02}:{minutes:02}:{seconds_int:02}.{milliseconds:03}"
        else:
            return f"{hours:02}:{minutes:02}:{seconds_int:02},{milliseconds:03}"

    def open_settings(self):
        settings_window = tk.Toplevel(self.master)
        settings_window.title("Settings")
        settings_window.geometry("450x500") # Increased height for better layout

        settings_window.grab_set() # Make settings window modal
        settings_window.transient(self.master)

        settings_frame = ttk.Frame(settings_window, padding="15")
        settings_frame.pack(fill=tk.BOTH, expand=True)

        # Hugging Face Token
        ttk.Label(settings_frame, text=self.translate("hf_token_label")).pack(anchor=tk.W, pady=(0, 2))
        self.hf_token_entry = ttk.Entry(settings_frame)
        self.hf_token_entry.insert(0, self.settings.get('huggingface_token', ''))
        self.hf_token_entry.pack(fill=tk.X, pady=(0, 10))

        # Output Directory
        ttk.Label(settings_frame, text=self.translate("output_directory_label")).pack(anchor=tk.W, pady=(0, 2))
        output_dir_frame = ttk.Frame(settings_frame)
        output_dir_frame.pack(fill=tk.X, pady=(0, 10))

        self.output_dir_entry = ttk.Entry(output_dir_frame)
        self.output_dir_entry.insert(0, self.settings.get('output_directory', os.getcwd()))
        self.output_dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(output_dir_frame, text=self.translate("browse_button"), command=self.browse_output_dir, bootstyle="secondary").pack(side=tk.RIGHT)

        # Save to input directory checkbox
        self.save_to_input_dir_var = tk.BooleanVar(value=self.settings.get('save_to_input_dir', False))
        ttk.Checkbutton(settings_frame, text=self.translate("save_to_input_dir_checkbox"), variable=self.save_to_input_dir_var).pack(anchor=tk.W, pady=(0, 10))

        # Output Formats (Checkboxes)
        ttk.Label(settings_frame, text=self.translate("output_format_label")).pack(anchor=tk.W, pady=(0, 2))
        
        self.output_formats_vars = {}
        available_formats = ["txt", "srt", "vtt"]
        current_selected_formats = self.settings.get('output_formats', ['txt']) # Default to txt

        for fmt in available_formats:
            var = tk.BooleanVar(value=(fmt in current_selected_formats))
            cb = ttk.Checkbutton(settings_frame, text=f".{fmt}", variable=var)
            cb.pack(anchor=tk.W, padx=20)
            self.output_formats_vars[fmt] = var
        
        ttk.Frame(settings_frame, height=10).pack() # Spacer

        # Re-align audio option
        self.realign_audio_var = tk.BooleanVar(value=self.settings.get('realign_audio', True))
        ttk.Checkbutton(settings_frame, text=self.translate("realign_audio_checkbox"), variable=self.realign_audio_var).pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(settings_frame, text=self.translate("realign_audio_description"), font=("TkDefaultFont", 8)).pack(anchor=tk.W, pady=(0, 10))

        # Chunk Size input
        ttk.Label(settings_frame, text=self.translate("chunk_size_label")).pack(anchor=tk.W, pady=(0, 2))
        self.chunk_size_entry = ttk.Entry(settings_frame)
        self.chunk_size_entry.insert(0, str(self.settings.get('chunk_size', 5))) # Default to 25
        self.chunk_size_entry.pack(fill=tk.X, pady=(0, 10))

        # Language selection for GUI
        ttk.Label(settings_frame, text=self.translate("gui_language_label")).pack(anchor=tk.W, pady=(0, 2))
        self.gui_language_combobox = ttk.Combobox(settings_frame, values=["English", "繁體中文"])
        
        # Set initial value based on current_language
        if self.current_language == 'en':
            self.gui_language_combobox.set("English")
        elif self.current_language == 'zh-TW':
            self.gui_language_combobox.set("繁體中文")
        else:
            self.gui_language_combobox.set("English") # Default to English

        self.gui_language_combobox.pack(fill=tk.X, pady=(0, 10))

        def save_and_close_settings():
            self.settings['huggingface_token'] = self.hf_token_entry.get()
            self.settings['output_directory'] = self.output_dir_entry.get()
            
            # Save selected output formats
            selected_formats = [fmt for fmt, var in self.output_formats_vars.items() if var.get()]
            self.settings['output_formats'] = selected_formats
            
            self.settings['save_to_input_dir'] = self.save_to_input_dir_var.get()
            self.settings['realign_audio'] = self.realign_audio_var.get() # Save the new setting
            
            try:
                chunk_size_val = int(self.chunk_size_entry.get())
                if chunk_size_val <= 0:
                    raise ValueError("Chunk size must be a positive integer.")
                self.settings['chunk_size'] = chunk_size_val
            except ValueError as e:
                messagebox.showerror("Invalid Input", f"Invalid chunk size: {e}")
                return # Do not save settings if input is invalid
            
            selected_gui_lang = self.gui_language_combobox.get()
            if selected_gui_lang == "English":
                self.settings['language'] = 'en'
            elif selected_gui_lang == "繁體中文":
                self.settings['language'] = 'zh-TW'
            
            self.save_settings()
            
            # Reload translations and update GUI if language changed
            if self.current_language != self.settings['language']:
                self.current_language = self.settings['language']
                self.translations = self.load_translations(self.current_language)
                self._apply_language()

            messagebox.showinfo(self.translate("settings_saved_info_title"), self.translate("settings_saved_info_message"))
            settings_window.destroy()

        ttk.Button(settings_frame, text=self.translate("save_settings_button"), command=save_and_close_settings, bootstyle="primary").pack(pady=10)

    def browse_output_dir(self):
        directory = filedialog.askdirectory(title=self.translate("select_output_directory_title"))
        if directory:
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, directory)

    def _apply_language(self):
        # Update main window title
        self.master.title(self.translate("app_title"))

        # Update file selection frame
        self.file_frame.config(text=self.translate("file_selection_frame_text"))
        self.add_files_button.config(text=self.translate("add_files_button"))
        self.delete_selected_files_button.config(text=self.translate("delete_selected_files_button")) # Update button name

        # Update options frame
        self.options_frame.config(text=self.translate("options_frame_text"))
        self.diarization_checkbox.config(text=self.translate("enable_diarization_checkbox"))
        self.language_label.config(text=self.translate("language_label"))
        self.model_label.config(text=self.translate("model_label"))
        self.initial_prompt_label.config(text=self.translate("initial_prompt_label")) # Update initial prompt label
        self.device_label.config(text=self.translate("device_label"))

        # Update action buttons
        self.transcribe_button.config(text=self.translate("start_transcription_button"))
        self.cancel_button.config(text=self.translate("cancel_transcription_button"))
        self.settings_button.config(text=self.translate("settings_button"))

        # Update output area
        self.output_frame.config(text=self.translate("output_area_frame_text"))

        # Update progress frame
        self.progress_frame.config(text=self.translate("progress_frame_text"))
        self.progress_label.config(text=self.translate("progress_label_ready"))

        # Update settings window elements if it's open (though it will be destroyed and recreated)
        # This part is mostly for consistency if the settings window were persistent.
        # For now, the settings window is recreated, so its elements will be translated on open.

if __name__ == "__main__":
    root = ttk.Window(themename="flatly") # Use a ttkbootstrap theme
    app = WhisperXGUI(root)
    root.mainloop()
