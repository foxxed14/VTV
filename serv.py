from flask import Flask, request, jsonify, send_file, url_for
import sqlite3
import speech_recognition as sr
import torch
from vosk import Model as VoskModel, KaldiRecognizer
import wave
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel, PeftConfig
from io import BytesIO
import soundfile as sf

app = Flask(__name__)
recognizer = sr.Recognizer()
vosk_model_path = r"models/vosk-model-small-ru-0.22"
vosk_model = VoskModel(vosk_model_path)  # путь к модели vosk
saiga_model_name = "IlyaGusev/saiga2_7b_lora"
base_model_path = "TheBloke/Llama-2-7B-fp16"
offload_dir = r"C:\Games\t1\offload_dir"

print("Загрузка моделей...")

tokenizer = AutoTokenizer.from_pretrained(saiga_model_name, use_fast=False)
print("Токенизатор загружен")

config = PeftConfig.from_pretrained(saiga_model_name)
saiga_model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,  # Изменено на float32
        device_map={"": "cpu"},  # Явное указание использования CPU
    ),
    saiga_model_name,
    torch_dtype=torch.float32,  # Изменено на float32
    device_map={"": "cpu"},  # Явное указание использования CPU
    offload_dir=offload_dir
)
generation_config = GenerationConfig.from_pretrained(saiga_model_name)
saiga_model.eval()
print("Модель Saiga2 загружена")

tts_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                              model='silero_tts',
                              language='ru',
                              speaker='v4_ru')
tts_model.to(torch.device('cpu'))
print("Модель Silero TTS загружена")

DB_PATH = 'database.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS responses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        question_text TEXT,
                        answer_text TEXT,
                        answer_audio BLOB
                    )''')
    conn.commit()
    conn.close()
    print("База данных инициализирована")

def add_to_db(question_text, answer_text, answer_audio):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO responses (question_text, answer_text, answer_audio) VALUES (?, ?, ?)', 
                   (question_text, answer_text, answer_audio))
    conn.commit()
    conn.close()
    print("Данные добавлены в базу")

def get_from_db(question_text):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT answer_text, answer_audio FROM responses WHERE question_text = ?', (question_text,))
    result = cursor.fetchone()
    conn.close()
    if result:
        print("Ответ найден в базе данных")
    else:
        print("Ответ не найден в базе данных")
    return result

def recognize_speech(audio_file_path):
    print(f"Распознавание речи в файле {audio_file_path}")
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio, language='ru-RU')
        print(f"Распознанный текст: {text}")
        return text
    except sr.UnknownValueError:
        print("Не удалось распознать речь")
        return None

def vosk_recognize(audio_file_path):
    print(f"Распознавание речи с помощью Vosk в файле {audio_file_path}")
    wf = wave.open(audio_file_path, "rb")
    rec = KaldiRecognizer(vosk_model, wf.getframerate())
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            break
    result = rec.Result()
    print(f"Распознанный текст Vosk: {result}")
    return result

def generate_response(prompt):
    print(f"Генерация ответа на запрос: {prompt}")
    data = tokenizer(prompt, return_tensors="pt")
    data = {k: v.to(saiga_model.device) for k, v in data.items()}
    output_ids = saiga_model.generate(
        **data,
        generation_config=generation_config
    )[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f"Сгенерированный ответ: {output}")
    return output.strip()

def silero_tts(text, model, speaker='aidar', sample_rate=48000, put_accent=True, put_yo=True):
    print(f"Конвертация текста в речь: {text}")
    audio = model.apply_tts(text=text, speaker=speaker, sample_rate=sample_rate, put_accent=put_accent, put_yo=put_yo)
    audio_bytes = BytesIO()
    sf.write(audio_bytes, audio.numpy(), sample_rate, format='wav')
    audio_bytes.seek(0)
    print("Текст конвертирован в речь")
    return audio_bytes.read()

@app.route('/upload', methods=['POST'])
def upload():
    audio_file = request.files['audio']
    audio_file_path = os.path.join("uploads", audio_file.filename)
    audio_file.save(audio_file_path)
    print(f"Аудиофайл сохранен: {audio_file_path}")

    recognized_text = recognize_speech(audio_file_path)
    if recognized_text:
        db_result = get_from_db(recognized_text)
        if db_result:
            print("Отправка ответа из базы данных")
            response_audio_path = os.path.join("uploads", "response.wav")
            with open(response_audio_path, "wb") as f:
                f.write(db_result[1])
            return jsonify({"answer_text": db_result[0], "answer_audio_url": url_for('get_audio', filename="response.wav", _external=True)})
        else:
            vosk_result = vosk_recognize(audio_file_path)
            prompt = f"<s>system\n Ты — Михаил, военный красной армии , сейчас 9 февраля 1943 года , Ты принимаешь участие в освобождении ростова и даешь все ответы от лица солдата красной армии, отвечаешь максимально просто и понятно как бы ответил офицер если его бы спросили о чем то во время боевых действий , раговариваешь на ты .\n</s>\n<s>user\n{vosk_result}\n</s>\n<s>bot\n"
            answer_text = generate_response(prompt)
            audio_bytes = silero_tts(answer_text, tts_model)
            add_to_db(vosk_result, answer_text, audio_bytes)
            response_audio_path = os.path.join("uploads", "response.wav")
            with open(response_audio_path, "wb") as f:
                f.write(audio_bytes)
            print("Отправка сгенерированного ответа")
            return jsonify({"answer_text": answer_text, "answer_audio_url": url_for('get_audio', filename="response.wav", _external=True)})
    else:
        print("Речь не распознана, ожидайте")
        return jsonify({"message": "ожидайте"})

@app.route('/upload', methods=['POST'])
def upload():
    audio_file = request.files['audio']
    audio_file_path = os.path.join("uploads", audio_file.filename)
    audio_file.save(audio_file_path)
    print(f"Аудиофайл сохранен: {audio_file_path}")

    recognized_text = recognize_speech(audio_file_path)
    if recognized_text:
        db_result = get_from_db(recognized_text)
        if db_result:
            print("Отправка ответа из базы данных")
            response_audio_path = os.path.join("uploads", "response.wav")
            with open(response_audio_path, "wb") as f:
                f.write(db_result[1])
            return jsonify({"answer_text": db_result[0], "answer_audio_url": url_for('get_audio', filename="response.wav", _external=True)})
        else:
            vosk_result = vosk_recognize(audio_file_path)
            prompt = f"<s>system\n Ты — Михаил, военный красной армии , сейчас 9 февраля 1943 года , Ты принимаешь участие в освобождении ростова и даешь все ответы от лица солдата красной армии, отвечаешь максимально просто и понятно как бы ответил офицер если его бы спросили о чем то во время боевых действий , раговариваешь на ты .\n</s>\n<s>user\n{vosk_result}\n</s>\n<s>bot\n"
            answer_text = generate_response(prompt)
            audio_bytes = silero_tts(answer_text, tts_model)
            add_to_db(vosk_result, answer_text, audio_bytes)
            response_audio_path = os.path.join("uploads", "response.wav")
            with open(response_audio_path, "wb") as f:
                f.write(audio_bytes)
            print("Отправка сгенерированного ответа")
            return jsonify({"answer_text": answer_text, "answer_audio_url": url_for('get_audio', filename="response.wav", _external=True)})
    else:
        print("Речь не распознана, ожидайте")
        return jsonify({"message": "ожидайте"})


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs(offload_dir, exist_ok=True)
    init_db()
    app.run(host='0.0.0.0', port=5000)
    print("Сервер запущен и готов к работе")
