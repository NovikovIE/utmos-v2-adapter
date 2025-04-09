from utmosv2 import create_model
import torch
import librosa

audio_tensor = torch.randn(50000)

model = create_model(pretrained=True)
preds = model.predict(input_path='_-a-c3-3.wav')

print(preds)

audio_t = librosa.load('_-a-c3-3.wav', sr=16000)
audio_t = audio_t[0]

preds = model.predict(
    input_tensor=audio_t,
    sample_rate=16000,
    device="cuda:0"
)

print(preds)

preds = model.predict(
    input_tensors=[audio_t, audio_t],
    sample_rate=16000,
    device="cuda:0"
)

print(preds)
