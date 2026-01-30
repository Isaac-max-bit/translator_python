import gradio as gr
import whisper
from translate import Translator
from dotenv import dotenv_values
from elevenlabs import ElevenLabs, VoiceSettings
import os
import uuid

# =========================
# Load environment variables
# =========================
config = dotenv_values(".env")
ELEVENLABS_API_KEY = config["ELEVENLABS_API_KEY"]

# =========================
# Load Whisper model ONCE
# =========================
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")

# =========================
# Translator function
# =========================
def translator(audio_file):
    try:
        # 1️⃣ Transcribe Spanish audio → text
        result = whisper_model.transcribe(audio_file, language="Spanish")
        transcription = result["text"].replace("\n", " ")

        # 2️⃣ Translate ES → EN
        en_transcription = Translator(
            from_lang="es",
            to_lang="en"
        ).translate(transcription)

        # 3️⃣ ElevenLabs client
        try:
            client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        except Exception as e:
            raise gr.Error(f"Failed to initialize ElevenLabs client: {str(e)}")

        # 4️⃣ Text to speech
        response = client.text_to_speech.convert(
            voice_id="pNInz6obpgDQGcFmaJgB",  # Adam
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=en_transcription,
            model_id="eleven_turbo_v2",
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
            ),
        )

        # 5️⃣ Ensure output dir exists
        os.makedirs("audios", exist_ok=True)

        # Unique filename (avoid overwrite)
        output_path = f"audios/en_{uuid.uuid4().hex}.mp3"

        # 6️⃣ Save audio stream
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        return output_path

    except Exception as e:
        raise gr.Error(f"Error processing audio: {str(e)}")

# =========================
# Gradio UI
# =========================
web = gr.Interface(
    fn=translator,
    inputs=gr.Audio(
        sources=["microphone"],
        type="filepath"
    ),
    outputs=gr.Audio(type="filepath"),
    title="🎙️ Audio Translator",
    description="Speak Spanish → Get English speech"
)

# =========================
# Launch app
# =========================
web.launch()
