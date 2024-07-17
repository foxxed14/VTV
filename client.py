import requests
from pydub import AudioSegment
import os

# URL сервера
SERVER_URL = "http://localhost:5000/upload"

# Функция для конвертации аудио файла в WAV
def convert_to_wav(audio_file_path, wav_file_path):
    print(f"Конвертация {audio_file_path} в {wav_file_path}")
    audio = AudioSegment.from_file(audio_file_path)
    audio.export(wav_file_path, format="wav")
    print(f"Файл конвертирован в {wav_file_path}")

# Функция отправки аудио файла на сервер и получения ответа
def send_audio_file(audio_file_path):
    # Конвертируем аудио файл в WAV
    wav_file_path = audio_file_path.rsplit(".", 1)[0] + ".wav"
    convert_to_wav(audio_file_path, wav_file_path)
    
    # Отправляем аудио файл на сервер
    print(f"Отправка {wav_file_path} на сервер")
    with open(wav_file_path, 'rb') as f:
        files = {'audio': f}
        response = requests.post(SERVER_URL, files=files)
    
    # Удаляем временный WAV файл
    os.remove(wav_file_path)
    print(f"Временный файл {wav_file_path} удален")
    
    if response.status_code == 200:
        data = response.json()
        text_response = data.get("answer_text", "")
        audio_url = data.get("answer_audio_url", "")
        
        if audio_url:
            print(f"Скачивание аудио ответа по URL: {audio_url}")
            audio_response = requests.get(audio_url)
            with open("response.wav", "wb") as f:
                f.write(audio_response.content)
            print("Ответ записан в response.wav")
        
        return text_response
    else:
        return "Ошибка: " + response.text

if __name__ == "__main__":
    audio_file_path = r"C:\Games\t1\1.m4a"  # путь к вашему MP3 или M4A файлу
    print(f"Отправка файла {audio_file_path}")
    text_response = send_audio_file(audio_file_path)
    print("Текстовый ответ:", text_response)
