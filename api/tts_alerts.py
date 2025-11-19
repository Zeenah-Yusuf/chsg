# api/tts_alerts.py
import asyncio
try:
    import edge_tts  # type: ignore
except Exception:
    # Lightweight fallback stub so the module can be imported in environments
    # where edge_tts is not available (e.g., editor/type checker).
    # This stub writes an empty placeholder file; replace with a real backend
    # for production use.
    class _EdgeTTSStub:
        class Communicate:
            def __init__(self, text, voice=None):
                self.text = text
                self.voice = voice

            async def save(self, output_file):
                # Create an empty file as a placeholder to avoid runtime errors.
                with open(output_file, "wb") as f:
                    f.write(b"")

    edge_tts = _EdgeTTSStub()

async def text_to_speech(text: str, voice: str = "en-US-AriaNeural", output_file: str = "alert.mp3"):
    """
    Convert text to speech using EdgeTTS.
    Voices:
      - Hausa: ha-NG-AminaNeural
      - Yoruba: yo-NG-LolaNeural
      - Igbo: ig-NG-ChineduNeural
    """
    communicate = edge_tts.Communicate(text, voice=voice)
    await communicate.save(output_file)
