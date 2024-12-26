######################################################
stage=0
stop_stage=3
PYTHON_ENVIRONMENT=distillw2n
python_version=3.10.12
######################################################
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Installing conda environment..."
    if conda info --envs | grep -q ${PYTHON_ENVIRONMENT}; then
        echo "Environment ${PYTHON_ENVIRONMENT} already exists."
    else
        conda create --name ${PYTHON_ENVIRONMENT} python=${python_version} --yes
    fi
    conda activate ${PYTHON_ENVIRONMENT}
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Installing dependencies..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    sudo apt-get update && apt-get install -y libsndfile1 ffmpeg
    pip install -r requirements.txt
fi