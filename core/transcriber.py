import nltk
import librosa
import torch
import json
import re
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# проверить пакеты nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


class MediaProcessor:
    def __init__(self, model_id="bond005/whisper-podlodka-turbo", device=None):
        """загрузить whisper."""
        self.target_sampling_rate = 16000
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Whisper model '{model_id}' on {self.device}...")
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(self.device)

    def transcribe(self, audio_path: str) -> str:
        """распознать аудио целиком и получить текст."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Аудиофайл '{audio_path}' не найден.")

        print(f"Loading audio '{audio_path}'...")
        audio_array, _ = librosa.load(audio_path, sr=self.target_sampling_rate, mono=True)
        print(f"Duration: {len(audio_array)/self.target_sampling_rate:.1f} sec")

        print("Transcribing...")
        inputs = self.processor(
            audio_array,
            sampling_rate=self.target_sampling_rate,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
        )
        inputs = inputs.to(self.device, self.model.dtype)

        generated_ids = self.model.generate(
            **inputs,
            return_timestamps=True,
            max_new_tokens=410,
            language='ru',
            task="transcribe",
            num_beams=5,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            condition_on_prev_tokens=False
        ) # вообще можно поменять этот конфиг. но был подобран такой по умолчанию

        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription.strip()

    def clean_text(self, text: str, bad_words_json_path: str) -> str:
        """заменить слова по json-файлу."""
        if not os.path.exists(bad_words_json_path):
            raise FileNotFoundError(f"Ошибка: JSON-файл словаря {bad_words_json_path} не найден.")

        try:
            with open(bad_words_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                replacements = {word: "***" for word in data}
            elif isinstance(data, dict):
                replacements = data
            else:
                raise ValueError("JSON-файл должен содержать либо список, либо словарь.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Ошибка структуры JSON: {e}")

        # сначала заменить длинные варианты
        sorted_keys = sorted(replacements.keys(), key=len, reverse=True)

        for word in sorted_keys:
            replacement = replacements[word]
            escaped_word = re.escape(word)
            # искать целые слова без учета регистра
            pattern = r'(?<!\w)' + escaped_word + r'(?!\w)'
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # убрать лишние пробелы
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)

        return text.strip()

    def process_full(self, audio_path: str, bad_words_json_path: str, output_path: str):
        """распознать, очистить и сохранить текст."""
        raw_text = self.transcribe(audio_path)
        cleaned_text = self.clean_text(raw_text, bad_words_json_path)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
            
        print(f"Goal complete. Output saved to: {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 3:
        audio = sys.argv[1]
        json_dict = sys.argv[2]
        out_txt = sys.argv[3]
        
        processor = MediaProcessor()
        processor.process_full(audio, json_dict, out_txt)
    else:
        print("Использование: python transcriber.py <аудиофайл> <bad_words.json> <output.txt>")
