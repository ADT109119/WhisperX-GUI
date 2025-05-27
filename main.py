import tkinter as tk
from tkinter import filedialog, messagebox
import os
import json
import threading
import torch
# import numpy as np # Not directly used in latest changes, can be removed if not needed by whisperx internals exposed

# Import ttkbootstrap for modern theme
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

# Import WhisperX and Pyannote
import whisperx
from pyannote.audio import Pipeline as DiarizationPipeline
# from pyannote.core import Segment # Not directly used

class WhisperXGUI:
    def __init__(self, master):
        self.master = master
        master.title("WhisperX GUI")
        master.geometry("900x900") 
        master.resizable(False, False) 

        self.diarization_pipeline = None 
        self.whisper_model = None 
        self._loaded_model_name = None 
        self._loaded_device = None 
        self.current_compute_type = None 
        self._transcription_cancelled = False 

        self.settings = self.load_settings()
        if 'output_directory' not in self.settings:
            self.settings['output_directory'] = os.path.join(os.getcwd(), "output")
        
        self.current_language = self.settings.get('language', 'en')
        self.translations = self.load_translations(self.current_language)
        self._apply_language_on_startup() 

        # ... (rest of __init__ remains the same) ...
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
        self.language_combobox = ttk.Combobox(self.options_frame, values=list(self.languages_map.keys()), state="readonly")
        initial_lang_code = self.settings.get('last_language', 'en')
        initial_lang_name = next((name for name, code in self.languages_map.items() if code == initial_lang_code), "English")
        self.language_combobox.set(initial_lang_name)
        self.language_combobox.pack(fill=tk.X, pady=2)

        self.model_label = ttk.Label(self.options_frame, text=self.translate("model_label"))
        self.model_label.pack(anchor=tk.W, pady=2)
        self.model_combobox = ttk.Combobox(self.options_frame, values=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo", "distil-large-v2"], state="readonly") 
        self.model_combobox.set(self.settings.get('last_model', 'base'))
        self.model_combobox.pack(fill=tk.X, pady=2)

        self.initial_prompt_label = ttk.Label(self.options_frame, text=self.translate("initial_prompt_label"))
        self.initial_prompt_label.pack(anchor=tk.W, pady=2)
        self.initial_prompt_entry = ttk.Entry(self.options_frame)
        self.initial_prompt_entry.insert(0, self.settings.get('initial_prompt', ''))
        self.initial_prompt_entry.pack(fill=tk.X, pady=2)

        self.device_label = ttk.Label(self.options_frame, text=self.translate("device_label"))
        self.device_label.pack(anchor=tk.W, pady=2)
        self.available_devices = ["CPU"]
        if torch.cuda.is_available():
            self.available_devices.append("GPU")
        self.device_combobox = ttk.Combobox(self.options_frame, values=self.available_devices, state="readonly")
        
        initial_device_name = self.settings.get('last_device', 'CPU')
        if initial_device_name not in self.available_devices:
            initial_device_name = 'CPU' 
        self.device_combobox.set(initial_device_name)
        self.device_combobox.pack(fill=tk.X, pady=2)

        self.button_frame = ttk.Frame(self.main_frame, padding="10")
        self.button_frame.pack(fill=tk.X, pady=10)

        self.transcribe_button = ttk.Button(self.button_frame, text=self.translate("start_transcription_button"), command=self.start_transcription, bootstyle="success")
        self.transcribe_button.pack(side=tk.LEFT, expand=True, padx=5)

        self.cancel_button = ttk.Button(self.button_frame, text=self.translate("cancel_transcription_button"), command=self.cancel_transcription, bootstyle="warning", state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT, expand=True, padx=5)

        self.settings_button = ttk.Button(self.button_frame, text=self.translate("settings_button"), command=self.open_settings, bootstyle="info")
        self.settings_button.pack(side=tk.RIGHT, expand=True, padx=5)

        self.output_frame = ttk.LabelFrame(self.main_frame, text=self.translate("output_area_frame_text"), padding="10")
        self.output_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.output_text_scrollbar = ttk.Scrollbar(self.output_frame, orient="vertical")
        self.output_text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.output_text = tk.Text(self.output_frame, wrap=tk.WORD, height=10, state=tk.DISABLED,
                                   yscrollcommand=self.output_text_scrollbar.set) 
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True) 

        self.output_text_scrollbar.config(command=self.output_text.yview)

        self.progress_frame = ttk.LabelFrame(self.main_frame, text=self.translate("progress_frame_text"), padding="10")
        self.progress_frame.pack(fill=tk.X, pady=10)
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient="horizontal", length=100, mode="determinate")
        self.progress_bar.pack(fill=tk.X, expand=True, pady=5)
        self.progress_label = ttk.Label(self.progress_frame, text=self.translate("progress_label_ready"))
        self.progress_label.pack(pady=2)


    def _apply_language_on_startup(self):
        self.master.title(self.translate("app_title"))

    def load_settings(self):
        settings_path = "settings.json"
        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r', encoding='utf-8') as f: 
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error: Could not decode {settings_path}. Using default settings.")
                return {}
            except Exception as e:
                print(f"Error loading settings: {e}. Using default settings.")
                return {}
        return {}

    def save_settings(self):
        settings_path = "settings.json"
        try:
            with open(settings_path, 'w', encoding='utf-8') as f: 
                json.dump(self.settings, f, indent=4, ensure_ascii=False)
        except Exception as e:
            messagebox.showerror(self.translate("settings_error_title"), f"{self.translate('settings_save_error_message')}: {e}")


    def load_translations(self, lang_code):
        lang_dir = "lang"
        if not os.path.exists(lang_dir):
            os.makedirs(lang_dir)
        lang_file = os.path.join(lang_dir, f"{lang_code}.json")
        default_lang_file = os.path.join(lang_dir, "en.json") 

        try:
            if os.path.exists(lang_file):
                with open(lang_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif os.path.exists(default_lang_file): 
                print(f"Warning: Language file {lang_file} not found. Falling back to English.")
                with open(default_lang_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else: 
                print(f"Critical Error: Default language file {default_lang_file} not found. Using keys as fallback.")
                # No messagebox here, as it might be too early for ttkbootstrap theming.
                # GUI will use keys as text.
                return {}
        except json.JSONDecodeError:
            messagebox.showerror("Language Error", f"Error decoding language file for {lang_code}. Check JSON syntax.")
            return {} 
        except Exception as e:
            messagebox.showerror("Language Error", f"Error loading language file for {lang_code}: {e}")
            return {}


    def translate(self, key):
        return self.translations.get(key, key) 

    def add_files(self):
        files = filedialog.askopenfilenames(
            title=self.translate("select_audio_video_files_title"),
            filetypes=[(self.translate("audio_video_files_filter"), "*.mp3 *.wav *.flac *.m4a *.mp4 *.avi *.mov *.ogg *.webm"), (self.translate("all_files_filter"), "*.*")]
        )
        for file in files:
            if file not in self.file_listbox.get(0, tk.END):
                self.file_listbox.insert(tk.END, file)

    def delete_selected_files(self):
        selected_indices = self.file_listbox.curselection()
        for i in reversed(selected_indices): 
            self.file_listbox.delete(i)

    def start_transcription(self):
        selected_files = self.file_listbox.get(0, tk.END)
        if not selected_files:
            messagebox.showwarning(self.translate("no_files_selected_title"), self.translate("no_files_selected_message"))
            return

        self.settings['last_language'] = self.languages_map.get(self.language_combobox.get(), 'en') 
        self.settings['last_model'] = self.model_combobox.get()
        self.settings['initial_prompt'] = self.initial_prompt_entry.get() 
        self.settings['last_device'] = self.device_combobox.get() 
        self.settings['diarization_enabled'] = self.diarization_var.get()
        self.save_settings()

        self.output_text.config(state=tk.NORMAL) 
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, self.translate("starting_transcription_message") + "\n")
        self.output_text.insert(tk.END, f"{self.translate('diarization_enabled_status')}: {self.diarization_var.get()}\n")
        self.output_text.insert(tk.END, f"{self.translate('language_status')}: {self.language_combobox.get()}\n")
        self.output_text.insert(tk.END, f"{self.translate('model_status')}: {self.model_combobox.get()}\n")
        self.output_text.insert(tk.END, f"{self.translate('initial_prompt_status')}: {self.initial_prompt_entry.get()}\n") 
        self.output_text.insert(tk.END, f"{self.translate('device_status')}: {self.device_combobox.get()}\n\n")
        self.output_text.config(state=tk.DISABLED) 

        self.transcribe_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL) 
        self.settings_button.config(state=tk.DISABLED)
        self.add_files_button.config(state=tk.DISABLED)
        self.delete_selected_files_button.config(state=tk.DISABLED) 
        self._transcription_cancelled = False 

        threading.Thread(target=self._run_transcription_process, args=(selected_files,)).start()

    def cancel_transcription(self):
        self._transcription_cancelled = True
        self.update_progress(0, 1, self.translate("cancelling_progress"))
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, self.translate("transcription_cancellation_requested") + "\n")
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
                    self.output_text.insert(tk.END, self.translate("transcription_cancelled_by_user") + "\n")
                    self.output_text.config(state=tk.DISABLED)
                    break 

                self.update_progress(i + 1, total_files, f"{self.translate('processing_file_progress')} {os.path.basename(file_path)}")
                self.output_text.config(state=tk.NORMAL)
                self.output_text.insert(tk.END, f"{self.translate('processing_file_message')} {i+1}/{total_files}: {os.path.basename(file_path)}\n")
                self.output_text.config(state=tk.DISABLED)
                self.master.update_idletasks()

                try:
                    save_to_input_dir = self.settings.get('save_to_input_dir', False)
                    transcribed_result_segments = self.transcribe_audio(file_path) 

                    if self._transcription_cancelled: 
                        self.output_text.config(state=tk.NORMAL)
                        self.output_text.insert(tk.END, self.translate("transcription_cancelled_during_processing") + "\n")
                        self.output_text.config(state=tk.DISABLED)
                        break

                    if not transcribed_result_segments: # If transcribe_audio returned None or empty
                        self.output_text.config(state=tk.NORMAL)
                        self.output_text.insert(tk.END, f"  {self.translate('no_segments_produced_for_file').format(file=os.path.basename(file_path))}\n\n")
                        self.output_text.config(state=tk.DISABLED)
                        self.master.update_idletasks()
                        continue # Skip to next file


                    base_filename = os.path.splitext(os.path.basename(file_path))[0]
                    if save_to_input_dir:
                        output_dir = os.path.dirname(file_path)
                    else:
                        default_output_dir = os.path.join(os.getcwd(), "output")
                        output_dir = self.settings.get('output_directory', default_output_dir)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir, exist_ok=True) 

                    selected_output_formats = self.settings.get('output_formats', ['txt']) 

                    for current_output_format in selected_output_formats:
                        output_filename = os.path.join(output_dir, f"{base_filename}.{current_output_format}")

                        if current_output_format == 'txt':
                            text_content = "\n".join([ (f"{s.get('speaker', self.translate('unknown_speaker_label_short'))}: " if 'speaker' in s else "") + s["text"] for s in transcribed_result_segments])
                            with open(output_filename, "w", encoding="utf-8") as f:
                                f.write(text_content)
                        elif current_output_format == 'srt':
                            srt_content = self._format_to_srt(transcribed_result_segments)
                            with open(output_filename, "w", encoding="utf-8") as f:
                                f.write(srt_content)
                        elif current_output_format == 'vtt':
                            vtt_content = self._format_to_vtt(transcribed_result_segments)
                            with open(output_filename, "w", encoding="utf-8") as f:
                                f.write(vtt_content)
                        
                        self.output_text.config(state=tk.NORMAL)
                        self.output_text.insert(tk.END, f"  {self.translate('transcription_saved_to')} {output_filename}\n")
                        self.output_text.config(state=tk.DISABLED)
                    self.output_text.config(state=tk.NORMAL)
                    self.output_text.insert(tk.END, "\n") 
                    self.output_text.config(state=tk.DISABLED)

                except Exception as e:
                    self.output_text.config(state=tk.NORMAL)
                    self.output_text.insert(tk.END, f"  {self.translate('error_processing_file')} {os.path.basename(file_path)}: {e}\n\n")
                    self.output_text.config(state=tk.DISABLED)
                    # Also print to console for easier debugging from terminal
                    import traceback
                    print(f"Error processing file {os.path.basename(file_path)}: {e}")
                    traceback.print_exc()
                self.master.update_idletasks()

            if not self._transcription_cancelled:
                self.update_progress(total_files, total_files, self.translate("transcription_complete_progress"))
                self.output_text.config(state=tk.NORMAL)
                self.output_text.insert(tk.END, self.translate("transcription_complete_message") + "\n")
                self.output_text.config(state=tk.DISABLED)
                messagebox.showinfo(self.translate("transcription_complete_title"), self.translate("all_files_processed_message"))
            else:
                self.update_progress(0, 1, self.translate("cancelled_progress_label"))
                messagebox.showinfo(self.translate("transcription_cancelled_title"), self.translate("transcription_process_was_cancelled_message"))


        except Exception as e:
            self.update_progress(0, 1, self.translate("error_occurred_progress"))
            self.output_text.config(state=tk.NORMAL)
            self.output_text.insert(tk.END, f"\n{self.translate('unexpected_error_during_processing')}: {e}\n")
            self.output_text.config(state=tk.DISABLED)
            messagebox.showerror(self.translate("error_title"), f"{self.translate('unexpected_error_occurred')}: {e}")
            import traceback
            print(f"Unexpected error during processing: {e}")
            traceback.print_exc()
        finally:
            self.transcribe_button.config(state=tk.NORMAL)
            self.cancel_button.config(state=tk.DISABLED) 
            self.settings_button.config(state=tk.NORMAL)
            self.add_files_button.config(state=tk.NORMAL)
            self.delete_selected_files_button.config(state=tk.NORMAL) 
            self.progress_label.config(text=self.translate("progress_label_ready"))

    def update_progress(self, value, maximum, text):
        self.progress_bar["value"] = value
        self.progress_bar["maximum"] = maximum
        self.progress_label.config(text=text)
        self.master.update_idletasks()

    def transcribe_audio(self, audio_path):
        # ... (model loading, ASR, alignment - same as before) ...
        model_name = self.model_combobox.get()
        language_display_name = self.language_combobox.get()
        language_code = self.languages_map.get(language_display_name, "en") 
        initial_prompt = self.initial_prompt_entry.get() 
        diarization_enabled = self.diarization_var.get()
        hf_token = self.settings.get('huggingface_token')
        selected_device_display_name = self.device_combobox.get()

        current_device = "cpu"
        self.current_compute_type = "int8"  

        if selected_device_display_name == "GPU" and torch.cuda.is_available():
            current_device = "cuda"
            self.current_compute_type = self.settings.get('gpu_compute_type', "float16") 
        else: 
            current_device = "cpu"
            self.current_compute_type = self.settings.get('cpu_compute_type', "int8") 
        
        model_dir = self.settings.get('model_cache_directory', os.path.join(os.getcwd(), "models")) 
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        self.update_progress(0, 100, self.translate("loading_whisper_model"))
        # Check for model reload conditions (name, device, compute_type, or language if not auto)
        asr_lang_param = language_code if language_code != "auto" else None
        current_model_lang = None
        if self.whisper_model and hasattr(self.whisper_model, 'model') and hasattr(self.whisper_model.model, 'config') and hasattr(self.whisper_model.model.config, 'language'):
            current_model_lang = self.whisper_model.model.config.language
        
        needs_reload = (
            self.whisper_model is None or
            self._loaded_model_name != model_name or
            self._loaded_device != current_device or
            (hasattr(self.whisper_model, 'compute_type') and self.whisper_model.compute_type != self.current_compute_type) or
            (asr_lang_param is not None and current_model_lang != asr_lang_param) # Reload if lang specified and different
        )

        if needs_reload:
            if self.whisper_model:
                del self.whisper_model
                if torch.cuda.is_available(): 
                    torch.cuda.empty_cache()
                self.whisper_model = None

            if self._transcription_cancelled:
                raise Exception(self.translate("transcription_cancelled_by_user_short"))
            
            self.whisper_model = whisperx.load_model(
                model_name,
                current_device, 
                compute_type=self.current_compute_type, 
                language=asr_lang_param,
                download_root=model_dir,
                asr_options = {"initial_prompt": initial_prompt}
            )
            self._loaded_model_name = model_name 
            self._loaded_device = current_device

        audio = whisperx.load_audio(audio_path)
        self.update_progress(20, 100, self.translate("transcribing_audio"))
        
        # For 'auto' language, WhisperX handles language detection internally if language=None in load_model.
        # If a specific language is set, it's passed to load_model.
        # We don't need to pass 'language' to transcribe() if it was passed to load_model.

        transcribe_verbose_str = str(self.settings.get('transcribe_verbose', "None")).lower()
        if transcribe_verbose_str == "true": transcribe_verbose_val = True
        elif transcribe_verbose_str == "false": transcribe_verbose_val = False
        else: transcribe_verbose_val = None


        transcribe_kwargs = {
            "batch_size": self.settings.get('batch_size', 16), # Default in whisperX is 16
            "verbose": transcribe_verbose_val,
            "chunk_size": self.settings.get('chunk_size', 30)
        }
        
        asr_result_dict = self.whisper_model.transcribe(audio, **transcribe_kwargs)
        segments_for_processing = asr_result_dict["segments"]
        detected_language = asr_result_dict["language"] 

        if language_code == "auto":
            detected_lang_name = detected_language.upper()
            for name, code in self.languages_map.items():
                if code == detected_language:
                    detected_lang_name = name
                    break
            self.output_text.config(state=tk.NORMAL)
            self.output_text.insert(tk.END, f"  {self.translate('language_detected_info')}: {detected_lang_name}\n")
            self.output_text.config(state=tk.DISABLED)


        if self.settings.get('realign_audio', True) and segments_for_processing: 
            self.update_progress(40, 100, self.translate("aligning_transcription"))
            align_language_code = detected_language 
            
            model_a, metadata = whisperx.load_align_model(language_code=align_language_code, device=current_device, model_dir=model_dir)

            if self._transcription_cancelled:
                del model_a
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                raise Exception(self.translate("transcription_cancelled_by_user_short"))

            aligned_segments_result = whisperx.align(
                segments_for_processing, 
                model_a,            
                metadata,           
                audio,              
                current_device,     
                return_char_alignments=False, 
            )
            segments_for_processing = aligned_segments_result["segments"] 
            del model_a 
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if not segments_for_processing: 
            self.output_text.config(state=tk.NORMAL)
            self.output_text.insert(tk.END, f"  {self.translate('no_speech_detected_in_file')}\n")
            self.output_text.config(state=tk.DISABLED)
            return [] 

        if diarization_enabled:
            if not hf_token:
                messagebox.showerror(self.translate("hf_token_missing_title"), self.translate("hf_token_required_error"))
                # Raise a specific error type or return None/empty to signal failure to _run_transcription_process
                return None # Signal error
            
            if self.diarization_pipeline is None or self.diarization_pipeline.device.type != current_device.split(':')[0]: 
                self.update_progress(60, 100, self.translate("loading_diarization_model"))
                if self._transcription_cancelled:
                    raise Exception(self.translate("transcription_cancelled_by_user_short"))
                
                if self.diarization_pipeline: 
                    del self.diarization_pipeline
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.diarization_pipeline = None

                pyannote_model_version = self.settings.get('pyannote_model_version', 'pyannote/speaker-diarization-3.1')
                self.diarization_pipeline = DiarizationPipeline.from_pretrained(
                    pyannote_model_version,
                    use_auth_token=hf_token, 
                    cache_dir=os.path.join(model_dir, "pyannote") 
                )
                self.diarization_pipeline.to(torch.device(current_device))

            self.update_progress(80, 100, self.translate("performing_speaker_diarization"))
            if self._transcription_cancelled:
                raise Exception(self.translate("transcription_cancelled_by_user_short"))

            num_speakers = self.settings.get('num_speakers', 0) 
            min_speakers = self.settings.get('min_speakers', 0) 
            max_speakers = self.settings.get('max_speakers', 0) 

            diarize_params = {}
            if num_speakers > 0: diarize_params['num_speakers'] = num_speakers
            if min_speakers > 0: diarize_params['min_speakers'] = min_speakers
            if max_speakers > 0: diarize_params['max_speakers'] = max_speakers
            
            # Pyannote expects a dict for audio input, with 'uri' and 'audio' (waveform)
            # or can take file path directly. Let's use file path if it's simpler.
            # audio_data_for_pyannote = {"waveform": torch.from_numpy(audio[None, :]), "sample_rate": whisperx.SAMPLE_RATE}
            # diarize_output_annotation = self.diarization_pipeline(audio_data_for_pyannote, **diarize_params)
            diarize_output_annotation = self.diarization_pipeline(audio_path, **diarize_params)


            # --- SPEAKER ASSIGNMENT PER WHISPER SEGMENT ---
            speaker_labeled_whisper_segments = []
            for seg_idx, whisper_segment in enumerate(segments_for_processing):
                # Basic validation of the whisper_segment
                if not isinstance(whisper_segment, dict) or \
                   whisper_segment.get("start") is None or \
                   whisper_segment.get("end") is None or \
                   not isinstance(whisper_segment.get("text", ""), str) or \
                   not whisper_segment.get("text", "").strip(): # Check if text is non-empty string after strip
                    self.output_text.config(state=tk.NORMAL)
                    self.output_text.insert(tk.END, f"  {self.translate('warning_skipping_invalid_asr_segment').format(index=seg_idx, content=str(whisper_segment)[:50])}\n")
                    self.output_text.config(state=tk.DISABLED)
                    self.master.update_idletasks()
                    continue

                segment_start = whisper_segment["start"]
                segment_end = whisper_segment["end"]
                
                if not (isinstance(segment_start, (int, float)) and isinstance(segment_end, (int, float)) and segment_start <= segment_end):
                    self.output_text.config(state=tk.NORMAL)
                    self.output_text.insert(tk.END, f"  {self.translate('warning_invalid_timestamp_asr_segment').format(index=seg_idx)}\n")
                    self.output_text.config(state=tk.DISABLED)
                    self.master.update_idletasks()
                    continue
                
                segment_midpoint = (segment_start + segment_end) / 2
                assigned_speaker = self.translate('unknown_speaker_label_short') 
                
                for turn, _, speaker_label in diarize_output_annotation.itertracks(yield_label=True):
                    if turn.start <= segment_midpoint <= turn.end:
                        assigned_speaker = speaker_label
                        break 
                
                speaker_labeled_whisper_segments.append({
                    "start": segment_start,
                    "end": segment_end,
                    "text": whisper_segment["text"].strip(),
                    "speaker": assigned_speaker
                    # "words": whisper_segment.get("words", []) # Keep words if alignment provided them
                })

            # --- CONDITIONAL MERGING ---
            # Default to False to prevent "糊成一坨"
            should_merge = self.settings.get('merge_diarized_segments', False) 

            if not should_merge:
                return speaker_labeled_whisper_segments
            else:
                # Apply the more aggressive merging logic if the user enabled it
                formatted_segments = []
                current_speaker = None
                current_text = ""
                current_start = None
                current_end = None

                for idx, segment_item in enumerate(speaker_labeled_whisper_segments):
                    speaker = segment_item.get("speaker", self.translate('unknown_speaker_label_short'))
                    text_content = segment_item.get("text")
                    start_time = segment_item.get("start")
                    end_time = segment_item.get("end")

                    # Redundant checks, but good for safety if segment_item is directly from elsewhere
                    if text_content is None or not text_content.strip(): continue
                    text = text_content.strip()
                    if start_time is None or end_time is None:
                        self.output_text.config(state=tk.NORMAL)
                        self.output_text.insert(tk.END, f"  {self.translate('warning_segment_missing_time_in_merge').format(index=idx, speaker=speaker, text=text[:30])}\n")
                        self.output_text.config(state=tk.DISABLED)
                        continue

                    if current_speaker is None: 
                        current_speaker = speaker
                        current_text = text
                        current_start = start_time
                        current_end = end_time
                    elif speaker == current_speaker and start_time <= (current_end + self.settings.get('speaker_merge_gap', 2.0)): 
                        current_text += " " + text
                        current_end = end_time 
                    else: 
                        if current_text: 
                            formatted_segments.append({
                                "speaker": current_speaker,
                                "text": current_text.strip(), 
                                "start": current_start,
                                "end": current_end
                            })
                        current_speaker = speaker
                        current_text = text
                        current_start = start_time
                        current_end = end_time
                
                if current_text: 
                    formatted_segments.append({
                        "speaker": current_speaker,
                        "text": current_text.strip(),
                        "start": current_start,
                        "end": current_end
                    })
                return formatted_segments
        else: # Not diarization_enabled
            clean_segments = []
            for segment_item in segments_for_processing:
                if not isinstance(segment_item, dict):
                    continue
                text_content = segment_item.get("text")
                # Ensure text is string and not just whitespace before adding
                if isinstance(text_content, str) and text_content.strip():
                    clean_segments.append({
                        "text": text_content.strip(), 
                        "start": segment_item.get("start"), 
                        "end": segment_item.get("end")
                        # "words": segment_item.get("words", [])
                    })
            return clean_segments

    def _format_to_srt(self, segments):
        srt_content = []
        for i, segment in enumerate(segments):
            if segment.get("start") is None or segment.get("end") is None or not segment.get("text","").strip(): continue 
            start_time = self._format_timestamp(segment["start"])
            end_time = self._format_timestamp(segment["end"])
            
            speaker_prefix = ""
            # Check if 'speaker' key exists and is not the default "UNKNOWN" placeholder before prepending
            # Also consider the setting 'show_unknown_speaker_label'
            raw_speaker = segment.get("speaker")
            if raw_speaker: # If speaker key exists
                if raw_speaker != self.translate('unknown_speaker_label_short'):
                     speaker_prefix = f"{raw_speaker}: "
                elif self.settings.get('show_unknown_speaker_label', True): # It is UNKNOWN, but user wants to see it
                     speaker_prefix = f"{self.translate('unknown_speaker_label')}: "
            
            text = speaker_prefix + segment["text"]
            srt_content.append(f"{i + 1}\n{start_time} --> {end_time}\n{text}\n")
        return "\n".join(srt_content)

    def _format_to_vtt(self, segments):
        vtt_content = ["WEBVTT\n"]
        for segment in segments:
            if segment.get("start") is None or segment.get("end") is None or not segment.get("text","").strip(): continue
            start_time = self._format_timestamp(segment["start"], vtt=True)
            end_time = self._format_timestamp(segment["end"], vtt=True)
            
            speaker_prefix = ""
            raw_speaker = segment.get("speaker")
            if raw_speaker:
                if raw_speaker != self.translate('unknown_speaker_label_short'):
                    speaker_display = raw_speaker # Could sanitize for VTT tags if needed
                    speaker_prefix = f"<v {speaker_display}>{speaker_display}</v>: "
                elif self.settings.get('show_unknown_speaker_label', True):
                    speaker_display = self.translate('unknown_speaker_label')
                    speaker_prefix = f"<v {speaker_display}>{speaker_display}</v>: "
            
            text = speaker_prefix + segment["text"]
            vtt_content.append(f"{start_time} --> {end_time}\n{text}\n")
        return "\n".join(vtt_content)

    def _format_timestamp(self, total_seconds, vtt=False):
        if total_seconds is None or not isinstance(total_seconds, (int, float)): 
            return "00:00:00,000" if not vtt else "00:00:00.000"
        
        # Ensure total_seconds is not negative, which can happen with erroneous data
        total_seconds = max(0, total_seconds)

        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds_int = int(total_seconds % 60)
        milliseconds = int(round((total_seconds - int(total_seconds)) * 1000)) 
        
        # Ensure milliseconds don't cause seconds_int to roll over due to rounding
        if milliseconds >= 1000:
            seconds_int += milliseconds // 1000
            milliseconds %= 1000
            if seconds_int >= 60:
                minutes += seconds_int // 60
                seconds_int %= 60
                if minutes >= 60:
                    hours += minutes // 60
                    minutes %= 60


        if vtt:
            return f"{hours:02}:{minutes:02}:{seconds_int:02}.{milliseconds:03}"
        else:
            return f"{hours:02}:{minutes:02}:{seconds_int:02},{milliseconds:03}"

    def open_settings(self):
        # ... (Settings window structure) ...
        # Add the new setting in the Diarization Tab:
        # self.merge_diarized_segments_var = tk.BooleanVar(value=self.settings.get('merge_diarized_segments', False)) # Default False
        # ttk.Checkbutton(diarization_tab, text=self.translate("merge_diarized_segments_checkbox"), variable=self.merge_diarized_segments_var).pack(anchor=tk.W, pady=(0, 10))
        # ...
        # In save_and_close_settings():
        # self.settings['merge_diarized_segments'] = self.merge_diarized_segments_var.get()
        # ...
        settings_window = tk.Toplevel(self.master)
        settings_window.title(self.translate("settings_window_title"))
        settings_window.geometry("500x800") # Increased height for new option

        settings_window.grab_set() 
        settings_window.transient(self.master)

        settings_frame = ttk.Frame(settings_window, padding="15")
        settings_frame.pack(fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(settings_frame)
        
        general_tab = ttk.Frame(notebook, padding="10")
        notebook.add(general_tab, text=self.translate("settings_tab_general"))
        # ... (General Tab content as before) ...
        ttk.Label(general_tab, text=self.translate("gui_language_label")).pack(anchor=tk.W, pady=(0, 2))
        self.gui_language_combobox = ttk.Combobox(general_tab, values=["English", "繁體中文"], state="readonly")
        if self.current_language == 'en': self.gui_language_combobox.set("English")
        elif self.current_language == 'zh-TW': self.gui_language_combobox.set("繁體中文")
        else: self.gui_language_combobox.set("English")
        self.gui_language_combobox.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(general_tab, text=self.translate("output_directory_label")).pack(anchor=tk.W, pady=(0, 2))
        output_dir_frame = ttk.Frame(general_tab)
        output_dir_frame.pack(fill=tk.X, pady=(0, 10))
        self.output_dir_entry = ttk.Entry(output_dir_frame)
        self.output_dir_entry.insert(0, self.settings.get('output_directory', os.path.join(os.getcwd(), "output")))
        self.output_dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(output_dir_frame, text=self.translate("browse_button"), command=self.browse_output_dir, bootstyle="secondary").pack(side=tk.RIGHT)

        self.save_to_input_dir_var = tk.BooleanVar(value=self.settings.get('save_to_input_dir', False))
        ttk.Checkbutton(general_tab, text=self.translate("save_to_input_dir_checkbox"), variable=self.save_to_input_dir_var).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(general_tab, text=self.translate("output_format_label")).pack(anchor=tk.W, pady=(0, 2))
        self.output_formats_vars = {}
        available_formats = ["txt", "srt", "vtt"] 
        current_selected_formats = self.settings.get('output_formats', ['txt']) 
        formats_frame = ttk.Frame(general_tab)
        formats_frame.pack(anchor=tk.W, pady=(0,10))
        for fmt in available_formats:
            var = tk.BooleanVar(value=(fmt in current_selected_formats))
            cb = ttk.Checkbutton(formats_frame, text=f".{fmt}", variable=var)
            cb.pack(side=tk.LEFT, padx=5) 
            self.output_formats_vars[fmt] = var

        ttk.Label(general_tab, text=self.translate("model_cache_directory_label")).pack(anchor=tk.W, pady=(0, 2))
        model_cache_dir_frame = ttk.Frame(general_tab)
        model_cache_dir_frame.pack(fill=tk.X, pady=(0, 10))
        self.model_cache_dir_entry = ttk.Entry(model_cache_dir_frame)
        self.model_cache_dir_entry.insert(0, self.settings.get('model_cache_directory', os.path.join(os.getcwd(), "models")))
        self.model_cache_dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(model_cache_dir_frame, text=self.translate("browse_button"), command=lambda: self.browse_model_cache_dir(self.model_cache_dir_entry), bootstyle="secondary").pack(side=tk.RIGHT)


        transcription_tab = ttk.Frame(notebook, padding="10")
        notebook.add(transcription_tab, text=self.translate("settings_tab_transcription"))
        # ... (Transcription Tab content as before) ...
        self.realign_audio_var = tk.BooleanVar(value=self.settings.get('realign_audio', True))
        ttk.Checkbutton(transcription_tab, text=self.translate("realign_audio_checkbox"), variable=self.realign_audio_var).pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(transcription_tab, text=self.translate("realign_audio_description"), font=("TkDefaultFont", 8), wraplength=400).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(transcription_tab, text=self.translate("chunk_size_label") + " (WhisperX default: 30)").pack(anchor=tk.W, pady=(0, 2))
        self.chunk_size_entry = ttk.Entry(transcription_tab)
        self.chunk_size_entry.insert(0, str(self.settings.get('chunk_size', 30))) 
        self.chunk_size_entry.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(transcription_tab, text=self.translate("batch_size_label") + " (WhisperX default: 16)").pack(anchor=tk.W, pady=(0, 2))
        self.batch_size_entry = ttk.Entry(transcription_tab)
        self.batch_size_entry.insert(0, str(self.settings.get('batch_size', 16)))
        self.batch_size_entry.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(transcription_tab, text=self.translate("no_speech_threshold_label") + " (Default: 0.6)").pack(anchor=tk.W, pady=(0,2))
        self.no_speech_threshold_entry = ttk.Entry(transcription_tab)
        self.no_speech_threshold_entry.insert(0, str(self.settings.get('no_speech_threshold', 0.6)))
        self.no_speech_threshold_entry.pack(fill=tk.X, pady=(0,10))
        
        self.transcribe_verbose_var = tk.StringVar(value=str(self.settings.get('transcribe_verbose', "None"))) 
        ttk.Label(transcription_tab, text=self.translate("transcribe_verbose_label")).pack(anchor=tk.W, pady=(0,2))
        verbose_options_frame = ttk.Frame(transcription_tab)
        verbose_options_frame.pack(anchor=tk.W, pady=(0,10))
        ttk.Radiobutton(verbose_options_frame, text="None", variable=self.transcribe_verbose_var, value="None").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(verbose_options_frame, text="True", variable=self.transcribe_verbose_var, value="True").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(verbose_options_frame, text="False", variable=self.transcribe_verbose_var, value="False").pack(side=tk.LEFT, padx=5)


        diarization_tab = ttk.Frame(notebook, padding="10")
        notebook.add(diarization_tab, text=self.translate("settings_tab_diarization"))
        # ... (Diarization Tab content as before) ...
        ttk.Label(diarization_tab, text=self.translate("hf_token_label")).pack(anchor=tk.W, pady=(0, 2))
        self.hf_token_entry = ttk.Entry(diarization_tab, show="*") 
        self.hf_token_entry.insert(0, self.settings.get('huggingface_token', ''))
        self.hf_token_entry.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(diarization_tab, text=self.translate("pyannote_model_version_label")).pack(anchor=tk.W, pady=(0,2))
        self.pyannote_model_combobox = ttk.Combobox(diarization_tab, values=["pyannote/speaker-diarization-3.1", "pyannote/speaker-diarization-2.1"], state="readonly")
        self.pyannote_model_combobox.set(self.settings.get('pyannote_model_version', 'pyannote/speaker-diarization-3.1'))
        self.pyannote_model_combobox.pack(fill=tk.X, pady=(0,10))

        ttk.Label(diarization_tab, text=self.translate("num_speakers_label") + " (0 for auto)").pack(anchor=tk.W, pady=(0,2))
        self.num_speakers_entry = ttk.Entry(diarization_tab)
        self.num_speakers_entry.insert(0, str(self.settings.get('num_speakers', 0)))
        self.num_speakers_entry.pack(fill=tk.X, pady=(0,10))
        
        ttk.Label(diarization_tab, text=self.translate("min_speakers_label") + " (0 for auto)").pack(anchor=tk.W, pady=(0,2))
        self.min_speakers_entry = ttk.Entry(diarization_tab)
        self.min_speakers_entry.insert(0, str(self.settings.get('min_speakers', 0)))
        self.min_speakers_entry.pack(fill=tk.X, pady=(0,10))

        ttk.Label(diarization_tab, text=self.translate("max_speakers_label") + " (0 for auto)").pack(anchor=tk.W, pady=(0,2))
        self.max_speakers_entry = ttk.Entry(diarization_tab)
        self.max_speakers_entry.insert(0, str(self.settings.get('max_speakers', 0)))
        self.max_speakers_entry.pack(fill=tk.X, pady=(0,10))
        
        # New Setting for merging diarized segments
        self.merge_diarized_segments_var = tk.BooleanVar(value=self.settings.get('merge_diarized_segments', False)) # Default False
        ttk.Checkbutton(diarization_tab, text=self.translate("merge_diarized_segments_checkbox"), variable=self.merge_diarized_segments_var).pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(diarization_tab, text=self.translate("merge_diarized_segments_description"), font=("TkDefaultFont", 8), wraplength=400).pack(anchor=tk.W, pady=(0, 10))


        ttk.Label(diarization_tab, text=self.translate("speaker_merge_gap_label") + " (seconds, Default: 2.0)").pack(anchor=tk.W, pady=(0,2))
        self.speaker_merge_gap_entry = ttk.Entry(diarization_tab)
        self.speaker_merge_gap_entry.insert(0, str(self.settings.get('speaker_merge_gap', 2.0)))
        self.speaker_merge_gap_entry.pack(fill=tk.X, pady=(0,10))
        
        self.show_unknown_speaker_label_var = tk.BooleanVar(value=self.settings.get('show_unknown_speaker_label', True))
        ttk.Checkbutton(diarization_tab, text=self.translate("show_unknown_speaker_label_checkbox"), variable=self.show_unknown_speaker_label_var).pack(anchor=tk.W, pady=(0, 10))


        advanced_tab = ttk.Frame(notebook, padding="10")
        notebook.add(advanced_tab, text=self.translate("settings_tab_advanced"))
        # ... (Advanced Tab content as before) ...
        ttk.Label(advanced_tab, text=self.translate("gpu_compute_type_label")).pack(anchor=tk.W, pady=(0,2))
        self.gpu_compute_type_combobox = ttk.Combobox(advanced_tab, values=["float16", "bfloat16", "float32", "int8_float16", "int8"], state="readonly") # Added more options
        self.gpu_compute_type_combobox.set(self.settings.get('gpu_compute_type', "float16"))
        self.gpu_compute_type_combobox.pack(fill=tk.X, pady=(0,10))
        
        ttk.Label(advanced_tab, text=self.translate("cpu_compute_type_label")).pack(anchor=tk.W, pady=(0,2))
        self.cpu_compute_type_combobox = ttk.Combobox(advanced_tab, values=["int8", "float32"], state="readonly")
        self.cpu_compute_type_combobox.set(self.settings.get('cpu_compute_type', "int8"))
        self.cpu_compute_type_combobox.pack(fill=tk.X, pady=(0,10))


        notebook.pack(expand=True, fill='both', pady=10)

        def save_and_close_settings():
            try:
                # General
                self.settings['output_directory'] = self.output_dir_entry.get()
                self.settings['model_cache_directory'] = self.model_cache_dir_entry.get()
                selected_formats = [fmt for fmt, var in self.output_formats_vars.items() if var.get()]
                if not selected_formats: 
                    messagebox.showwarning(self.translate("settings_warning_title"), self.translate("settings_no_output_format_selected"))
                    selected_formats = ['txt'] 
                self.settings['output_formats'] = selected_formats
                self.settings['save_to_input_dir'] = self.save_to_input_dir_var.get()
                
                # Transcription
                self.settings['realign_audio'] = self.realign_audio_var.get()
                self.settings['chunk_size'] = int(self.chunk_size_entry.get())
                self.settings['batch_size'] = int(self.batch_size_entry.get())
                self.settings['no_speech_threshold'] = float(self.no_speech_threshold_entry.get())
                
                verbose_val_str = self.transcribe_verbose_var.get()
                if verbose_val_str == "True": self.settings['transcribe_verbose'] = True
                elif verbose_val_str == "False": self.settings['transcribe_verbose'] = False
                else: self.settings['transcribe_verbose'] = None

                # Diarization
                self.settings['huggingface_token'] = self.hf_token_entry.get()
                self.settings['pyannote_model_version'] = self.pyannote_model_combobox.get()
                self.settings['num_speakers'] = int(self.num_speakers_entry.get())
                self.settings['min_speakers'] = int(self.min_speakers_entry.get())
                self.settings['max_speakers'] = int(self.max_speakers_entry.get())
                self.settings['merge_diarized_segments'] = self.merge_diarized_segments_var.get() # Save new setting
                self.settings['speaker_merge_gap'] = float(self.speaker_merge_gap_entry.get())
                self.settings['show_unknown_speaker_label'] = self.show_unknown_speaker_label_var.get()

                # Advanced
                self.settings['gpu_compute_type'] = self.gpu_compute_type_combobox.get()
                self.settings['cpu_compute_type'] = self.cpu_compute_type_combobox.get()

                selected_gui_lang = self.gui_language_combobox.get()
                new_lang_code = 'en' 
                if selected_gui_lang == "English": new_lang_code = 'en'
                elif selected_gui_lang == "繁體中文": new_lang_code = 'zh-TW'
                
                lang_changed = self.current_language != new_lang_code
                if lang_changed:
                    self.settings['language'] = new_lang_code 
                
                self.save_settings() 
                
                if lang_changed:
                    messagebox.showinfo(self.translate("language_change_title"), self.translate("language_change_message_restart")) # Changed message
                    self.current_language = new_lang_code # Update runtime language
                    self.translations = self.load_translations(self.current_language) # Reload translations
                    self._apply_language() # Apply to main window
                    # For settings window, it's better to destroy and reopen if language changes, or advise restart.
                    # Since we are closing it anyway, the next open will use new translations.
                
                messagebox.showinfo(self.translate("settings_saved_info_title"), self.translate("settings_saved_info_message"))
                settings_window.destroy()

            except ValueError as e:
                messagebox.showerror(self.translate("invalid_input_title"), f"{self.translate('invalid_input_message')}: {e}")
            except Exception as e:
                messagebox.showerror(self.translate("error_title"), f"{self.translate('error_saving_settings')}: {e}")


        ttk.Button(settings_frame, text=self.translate("save_settings_button"), command=save_and_close_settings, bootstyle="primary").pack(pady=10, side=tk.RIGHT)
        ttk.Button(settings_frame, text=self.translate("cancel_button"), command=settings_window.destroy, bootstyle="secondary").pack(pady=10, side=tk.RIGHT, padx=5)


    def browse_output_dir(self): 
        directory = filedialog.askdirectory(title=self.translate("select_output_directory_title"))
        if directory:
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, directory)

    def browse_model_cache_dir(self, entry_widget): 
        directory = filedialog.askdirectory(title=self.translate("select_model_cache_directory_title"))
        if directory:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, directory)


    def _apply_language(self):
        self.master.title(self.translate("app_title"))

        self.file_frame.config(text=self.translate("file_selection_frame_text"))
        self.add_files_button.config(text=self.translate("add_files_button"))
        self.delete_selected_files_button.config(text=self.translate("delete_selected_files_button")) 

        self.options_frame.config(text=self.translate("options_frame_text"))
        self.diarization_checkbox.config(text=self.translate("enable_diarization_checkbox"))
        self.language_label.config(text=self.translate("language_label"))
        self.model_label.config(text=self.translate("model_label"))
        self.initial_prompt_label.config(text=self.translate("initial_prompt_label")) 
        self.device_label.config(text=self.translate("device_label"))

        self.transcribe_button.config(text=self.translate("start_transcription_button"))
        self.cancel_button.config(text=self.translate("cancel_transcription_button"))
        self.settings_button.config(text=self.translate("settings_button"))

        self.output_frame.config(text=self.translate("output_area_frame_text"))
        self.progress_frame.config(text=self.translate("progress_frame_text"))
        
        current_progress_text = self.progress_label.cget("text")
        ready_text_key = "progress_label_ready"
        # Check if current text is one of the "idle" messages before changing it
        is_idle = False
        for lang_translations in [self.translations, self.load_translations('en'), self.load_translations('zh-TW')]: # Check against all known "ready" texts
            if current_progress_text == lang_translations.get(ready_text_key, ready_text_key):
                is_idle = True
                break
        if is_idle or current_progress_text == "Ready": # Fallback for untranslated key
             self.progress_label.config(text=self.translate(ready_text_key))


if __name__ == "__main__":
    root = ttk.Window(themename="flatly") # Use a ttkbootstrap theme
    app = WhisperXGUI(root)
    root.mainloop()
