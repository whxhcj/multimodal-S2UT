import faulthandler
faulthandler.enable()

import torch, torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"

# model_dir = "/opt/data/private/dsy/project/checkpoint/wav2vec2-base-10k-voxpopuli-ft-es"
model_dir = "/opt/data/private/dsy/project/checkpoint/wav2vec2-large-xlsr-53-spanish"
processor = Wav2Vec2Processor.from_pretrained(model_dir)
model = Wav2Vec2ForCTC.from_pretrained(model_dir)
model.to(device)

audio_path = "/opt/data/private/dsy/project/dataset/multi30k-dataset/data/speech/16khz_wav/es/valid/1.wav"
audio = torchaudio.load(audio_path)[0]#.squeeze(0)

# input_values = processor(audio, return_tensors="pt").input_values
input_values = audio
input_values = input_values.to(device)
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)

transcription = processor.decode(predicted_ids[0])
print(transcription)