import streamlit as st
from core import generate
import sounddevice as sd
import numpy as np
import wave
import core 

st.title("🎤 Can you get 5 stars on this level?")

text = generate()
st.subheader(f"Say: {text}")

def record_audio():
    st.write("🎙️ Let's Practice!")

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

if st.button("Start Recording"):
    audio_file = record_audio()
    st.audio(audio_file, format="audio/wav")

    transcription = core.transcription_func(audio_file)
    st.subheader("📝 Transcription:")
    st.write(transcription)

    wer_score = core.evaluation(transcription, text)
    st.subheader("📊 Your Score:")
    # st.write(f"{wer_score:.2f}%")
    
    star = "⭐"
    if(wer_score > 80):
        st.write(f"Great Job! {star, star, star, star, star}")
    elif(wer_score > 60):
        st.write(f"Nice Job! {star, star, star, star}")
    elif(wer_score > 40):
        st.write(f"Nice Job! {star, star, star}")
    elif(wer_score > 20):
        st.write(f"Nice Job! {star, star}")
    else:
        st.write(f"Nice Job! {star}")
