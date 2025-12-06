# cd filelists/LibriTTS && \
# ln -s /autodl-fs/data/LibriTTS/train-clean-100 train-clean-100 && \
# ln -s /data/ssd1/tianyi.tan/LibriTTS/train-clean-360 train-clean-360 && \
# ln -s /data/ssd1/tianyi.tan/LibriTTS/train-other-500 train-other-500 && \
# ln -s /autodl-fs/data/LibriTTS/dev-clean dev-clean && \
# ln -s /autodl-fs/data/LibriTTS/dev-other dev-other && \
# ln -s /data/ssd1/tianyi.tan/LibriTTS/test-clean test-clean && \
# ln -s /data/ssd1/tianyi.tan/LibriTTS/test-other test-other && \
# cd ../..

python train_tensorboard.py \
--config configs/sawsinsub_16khz.json \
--input_wavs_dir /data1/tizzytan/LibriTTS \
--input_training_file filelists/LibriTTS/train-clean.txt \
--input_validation_file filelists/LibriTTS/val-full.txt \
--list_input_unseen_wavs_dir /data1/tizzytan/LibriTTS /data1/tizzytan/LibriTTS \
--list_input_unseen_validation_file filelists/LibriTTS/dev-clean.txt filelists/LibriTTS/dev-other.txt \
--checkpoint_path /data1/tizzytan/exp/sawsinsub_16khz_80band
