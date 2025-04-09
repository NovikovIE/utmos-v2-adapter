from utmosv2 import create_model
import torch

audio_tensor = torch.randn(50000)

model = create_model(pretrained=True)
preds = model.predict(input_path='_-a-c3-2.wav')

print(preds)

preds = model.predict(
    input_tensor=audio_tensor,
    sample_rate=16000,
    device="cuda:0"
)

print(preds)

preds = model.predict(
    input_tensors=[audio_tensor, audio_tensor],
    sample_rate=16000,
    device="cuda:0"
)

print(preds)
