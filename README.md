# UTMOSv2 PyTorch Inference

It's adapted https://github.com/sarulab-speech/UTMOSv2 repo for inferencing utmosv2 model as `model.predict(torch.tensor)` or `model.predict(np.ndarray)`

example usage in `example.py`

```python
audio, sr = librosa.load(audio_file, sr=16000)

preds = model.predict(
    input_tensor=audio,
    sample_rate=sr,
    device="cuda:0"
)

# or multiple inputs

preds = model.predict(
    input_tensors=[audio, audio],
    sample_rate=sr,
    device="cuda:0"
)
```