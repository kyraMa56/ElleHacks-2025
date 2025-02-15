import streamlit as st
import sounddevice as sd
import numpy as np
import wave
import core 

def record_audio():
    st.write("🎙️ Speech Practise")

    SAMPLE_RATE = 16000 
    CHANNELS = 1
    DURATION = 5
    
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.int16)
    sd.wait()
    
    # Save as WAV file
    file_path = "recordings/recorded_audio.wav"
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())

    st.success("✅ Recording saved!")
    return file_path

st.title("🎤 Real-Time Speech-to-Text with Whisper")

if st.button("Start Recording"):
    audio_file = record_audio()
    st.audio(audio_file, format="audio/wav")

    transcription = core.transcription_func(audio_file)
    st.subheader("📝 Transcription:")
    st.write(transcription)

    wer_score = core.evaluation(transcription)
    st.subheader("📊 WER Score:")
    st.write(f"{wer_score:.2f}%")