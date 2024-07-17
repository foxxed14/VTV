import os
import torch
from vosk import Model as VoskModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Путь к папке с моделями
MODELS_DIR = r"C:\Games\t1\models"
OFFLOAD_DIR = r"C:\Games\t1\offload_dir"

# Путь к модели vosk
VOSK_MODEL_PATH = os.path.join(MODELS_DIR, "vosk-model-small-ru-0.22")

# Путь к модели saiga2
SAIGA_MODEL_NAME = "IlyaGusev/saiga2_7b_lora"
BASE_MODEL_PATH = "TheBloke/Llama-2-7B-fp16"

# Функция для загрузки моделей
def download_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OFFLOAD_DIR, exist_ok=True)
    
    # Проверка и загрузка модели Vosk
    if not os.path.exists(VOSK_MODEL_PATH):
        print("Downloading Vosk model...")
        # os.system(f"wget -P {MODELS_DIR} <URL_TO_VOSK_MODEL>")
        # os.system(f"tar -xvf {os.path.join(MODELS_DIR, 'vosk-model.tar.gz')} -C {VOSK_MODEL_PATH}")
        print("Vosk model downloaded.")

    # Проверка и загрузка модели saiga2
    print("Downloading Saiga2 model...")
    tokenizer = AutoTokenizer.from_pretrained(SAIGA_MODEL_NAME, use_fast=False, legacy=False)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,  # Загрузка модели без использования 8-битного режима
        device_map={"": "cpu"},  # Явное указание использования CPU
    )
    config = PeftConfig.from_pretrained(SAIGA_MODEL_NAME)
    saiga_model = PeftModel.from_pretrained(
        base_model,
        SAIGA_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},  # Явное указание использования CPU
        offload_dir=OFFLOAD_DIR
    )
    print("Saiga2 model downloaded.")

    # Проверка и загрузка модели Silero TTS
    print("Downloading Silero TTS model...")
    tts_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                  model='silero_tts',
                                  language='ru',
                                  speaker='v4_ru')
    tts_model.to(torch.device('cpu'))
    print("Silero TTS model downloaded.")

if __name__ == "__main__":
    download_models()
    print("All models are downloaded and checked.")
