# DistillW2N

The official codes for our paper: for "DistillW2N: A Lightweight One-Shot Whisper to Normal Voice Conversion Model Using Distillation of Self-Supervised Features" (DistillW2N), which is accepted by ICASSP2025. We referred to the implementations of [SoundStream](https://github.com/kaiidams/soundstream-pytorch) to build up the repository.

![system_diagram](https://github.com/user-attachments/assets/929662bf-263a-4d50-bc43-1f2ea66de777)

## Preparation
### Environments
We recommended the following dependencies:

- python >= 3.8
- torch >= 1.12.0
- torchaudio >= 0.13.0
  
### Datasets
You just need to download the datasets under `YOURPATH`.

## Training
```Shell
python u2ss2u.py
```
