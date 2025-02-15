import json
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
from huggingface_hub import InferenceClient
import torch
from evaluate import load
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

with open('hf_api_key.json', 'r') as file:
    data = json.load(file)
API_TOKEN = data["HF_API_KEY"]

inference = InferenceClient(model="openai/whisper-small", token=API_TOKEN)
processor = AutoProcessor.from_pretrained("stringbot/whisper-small-hi")
model = AutoModelForSpeechSeq2Seq.from_pretrained("stringbot/whisper-small-hi")
# processor = WhisperProcessor.from_pretrained("openai/whisper-small")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None

def transcription():
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)
    sample = ds[0]["audio"]
    input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

    # generate token ids
    predicted_ids = model.generate(input_features)
    # decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    return transcription

def map_to_pred(batch):
    audio = batch["audio"]
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
    batch["reference"] = processor.tokenizer._normalize(batch['text'])

    with torch.no_grad():
        predicted_ids = model.generate(input_features.to("cuda"))[0]
    transcription = processor.decode(predicted_ids)
    batch["prediction"] = processor.tokenizer._normalize(transcription)
    return batch

def evaluation(output):
    librispeech_test_clean = load_dataset("librispeech_asr", "clean", split="test")
    result = librispeech_test_clean.map(map_to_pred)

    wer = load("wer")
    score = 100 * wer.compute(references=result["reference"], predictions=result["prediction"])

    return score

if __name__ == '__main__':
    '''
    [' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.']
    '''
    print(transcription())
    