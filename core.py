import json
import time
import torch
from flask import Flask
import pyaudio
import numpy as np
import wave 
from flask import Flask
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from datasets import load_dataset
from huggingface_hub import InferenceClient
from evaluate import load
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

with open('hf_api_key.json', 'r') as file:
    data = json.load(file)
API_TOKEN = data["HF_API_KEY"]

inference = InferenceClient(model="openai/whisper-small", token=API_TOKEN)
# processor = AutoProcessor.from_pretrained("stringbot/whisper-small-hi")
# model = AutoModelForSpeechSeq2Seq.from_pretrained("stringbot/whisper-small-hi")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None

def transcription():
    # Taking audio input from user
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)
    sample = ds[0]["audio"]
    input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

    # generate token ids
    predicted_ids = model.generate(input_features)
    # decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    return transcription

def real_time_transcription(press_record=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=device)

    # when user press record
    if press_record:
        # Audio Settings
        FORMAT = pyaudio.paInt16  # 16-bit format
        CHANNELS = 1  # Mono
        RATE = 16000  # Sample rate
        CHUNK = 1024  # Buffer size
        RECORD_SECONDS = 5  
        TOTAL_DURATION = 10

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

        print("Listening...")
        start_time = time.time()
        while time.time() - start_time < TOTAL_DURATION:
            frames = []
            
            # Record short audio chunks
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            
            # Convert audio to numpy array
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0
            
            transcription = asr_pipeline(audio_data)
            print("Transcription:", transcription["text"])
            
    p.terminate()
    stream.stop_stream()
    stream.close()
    print('End transcription')
    
    return transcription["text"]

def map_to_pred(batch):
    audio = batch["audio"]
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
    batch["reference"] = processor.tokenizer._normalize(batch['text'])

    with torch.no_grad():
        predicted_ids = model.generate(input_features.to("cuda"))[0]
    transcription = processor.decode(predicted_ids)
    batch["prediction"] = processor.tokenizer._normalize(transcription)
    return batch

def evaluation():
    # Create new testing dataset/references
    # user_input =  [real_time_transcription(press_record=True)]
    user_input = ['I have four pens']
    references = ['I have four pencils']
    
    wer = load("wer")
    score = 100 * wer.compute(references=references, predictions=user_input)
    
    return score

if __name__ == '__main__':
    app = Flask(__name__)
    @app.route('/output')
    
    def get_output():
        return {"output": evaluation()}
    

