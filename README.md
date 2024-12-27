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
- Dataset Download
  - For the libritts, ljspeech, and timit datasets, [datahelper](https://github.com/tan90xx/distillw2n/tree/master/datahelper) will automatically download if they are not found at `YOURPATH`.
  - For the wtimit dataset, you will need to request it via email. Follow the appropriate procedures to obtain access and download the dataset to `YOURPATH`.
- Dataset Preparation (Option)
  - The [datapreper](https://github.com/tan90xx/distillw2n/tree/master/datapreper) offers options for PPW (Pseudo-whisper) and VAD (Voice Activity Detection) versions. You can choose to apply these processing steps according to your project's requirements.

## Inference
For quickvc and wesper please run:
```Shell
python compare_infer.py
```
For our models please run:
```Shell
python infer.py
```

## Training
```Shell
python u2ss2u.py
```
