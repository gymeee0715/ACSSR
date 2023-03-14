from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch
import torchaudio
'''
# inference
'''

# load model and processor
processor = Wav2Vec2Processor.from_pretrained("pretrained_weight")
model = Wav2Vec2ForCTC.from_pretrained("pretrained_weight")

# load speech
speech_array, sampling_rate = torchaudio.load("waves/ses1_spk1_L2_0.560_1.560.wav")
# tokenize
input_values = processor(speech_array[0], return_tensors="pt", padding="longest").input_values  # Batch size 1

# retrieve logits
logits = model(input_values).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)

print(transcription)

