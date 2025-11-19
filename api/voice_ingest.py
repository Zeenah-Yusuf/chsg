# api/voice_ingest.py
from transformers import pipeline

# Load Whisper multilingual model
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")

def transcribe_audio(file_path: str, language: str = None) -> str:
    """
    Transcribe audio file to text using Whisper.
    language can be 'ha' (Hausa), 'yo' (Yoruba), 'ig' (Igbo).
    """
    if language:
        result = asr(file_path, generate_kwargs={"language": language})
    else:
        result = asr(file_path)
    return result["text"]
