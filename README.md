# DistillW2N

The official codes for our paper: for "DistillW2N: A Lightweight One-Shot Whisper to Normal Voice Conversion Model Using Distillation of Self-Supervised Features" (DistillW2N), which is accepted by ICASSP2025. We referred to the implementations of [SoundStream](https://github.com/kaiidams/soundstream-pytorch) to build up the repository.

![system_diagram](https://github.com/user-attachments/assets/929662bf-263a-4d50-bc43-1f2ea66de777)

## Preparation
### Environments
```Shell
# create virtual python environment
conda create --name distillw2n python=3.10.12 --yes

# install dependencies
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
sudo apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install -r requirements.txt
```
  
### Datasets
You just need to download the datasets under `YOURPATH`.

## Training
```Shell
python u2ss2u.py
```
