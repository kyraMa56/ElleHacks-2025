import os
import json
import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from datasets import load_dataset
from huggingface_hub import InferenceClient
from evaluate import load
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

os.environ["PATH"] += os.pathsep + "C:/ffmpeg/bin" 

with open('hf_api_key.json', 'r') as file:
    data = json.load(file)
API_TOKEN = data["HF_API_KEY"]

inference = InferenceClient(model="openai/whisper-small", token=API_TOKEN)
# processor = AutoProcessor.from_pretrained("stringbot/whisper-small-hi")
# model = AutoModelForSpeechSeq2Seq.from_pretrained("stringbot/whisper-small-hi")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None

def transcription_func(audio_data):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=device)
    
    transcription = asr_pipeline(audio_data)
    print("Transcription:", transcription["text"])
            
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

def evaluation(transcription):
    # Create new testing dataset/references
    # user_input =  [real_time_transcription(press_record=True)]
    user_input = [transcription]
    references = ['I have four pencils']
    
    wer = load("wer")
    score = 100 * wer.compute(references=references, predictions=user_input)
    
    return score

if __name__ == '__main__':
    print('score: ',evaluation())
    

