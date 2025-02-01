# DistillW2N

The official codes for our paper: for "DistillW2N: A Lightweight One-Shot Whisper to Normal Voice Conversion Model Using Distillation of Self-Supervised Features" (DistillW2N), which is accepted by ICASSP2025.
![system_diagram](https://github.com/user-attachments/assets/929662bf-263a-4d50-bc43-1f2ea66de777)

## Setup
1. Create a Python environment with e.g. conda: `conda create --name distillw2n python=3.10.12 --yes`
2. Activate the new environment: `conda activate distillw2n`
3. Install torch and torchaudio: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121`
4. Update the packages: `sudo apt-get update && apt-get install -y libsndfile1 ffmpeg`
5. Install requirements with `pip install -r requirements.txt`
6. Download models with links given in [txt](https://github.com/tan90xx/distillw2n/blob/master/experiments/)

## Inference
- For quickvc and wesper please run: `python compare_infer.py`
- For our models please run: `python infer.py`

## Training
- Please run: `python u2ss2u.py`

## Datasets
You just need to download the datasets under `YOURPATH`. 
- Dataset Download
  - For the libritts, ljspeech, and timit datasets, [datahelper](https://github.com/tan90xx/distillw2n/tree/master/datahelper) will automatically download if they are not found at `YOURPATH`.
  - For the wtimit dataset, you will need to request it via email. Follow the appropriate procedures to obtain access and download the dataset to `YOURPATH`.
- Dataset Preparation (Option)
  - [datapreper](https://github.com/tan90xx/distillw2n/tree/master/datapreper) offers options for ppw (Pseudo-whisper) and vad (Voice Activity Detection) versions. You can choose to apply these processing steps according to your project's requirements.

## Credits
This implementation builds on
- [SoundStream](https://github.com/kaiidams/soundstream-pytorch) for the training pipeline.
