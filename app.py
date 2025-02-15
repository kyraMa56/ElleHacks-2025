import streamlit as st
import sounddevice as sd
import wave
import core 

def record_audio():
    st.write("🎙️ Recording for 5 seconds...")

    CHANNELS = 1  # Mono
    RATE = 16000  # Sample rate
    CHUNK = 1024  # Buffer size
    RECORD_SECONDS = 5  
    TOTAL_DURATION = 10
    
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.int16)
    sd.wait()
    
    # Save as WAV file
    file_path = "recorded_audio.wav"
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

    # Transcription
    transcription = transcribe_audio(audio_file)
    st.subheader("📝 Transcription:")
    st.write(transcription)

    # WER Score
    wer_score = evaluate_transcription(transcription)
    st.subheader("📊 WER Score:")
    st.write(f"{wer_score:.2f}%")