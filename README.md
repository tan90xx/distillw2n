# distillw2n-demo

Demo page for "DistillW2N: A Lightweight One-Shot Whisper to Normal Voice Conversion Model Using Distillation of Self-Supervised Features" (DistillW2N)

[https://github.com/tan90xx/distillw2n](https://github.com/tan90xx/distillw2n)

⚠️ Our Token2Wav "vocoder" was trained using **less than 100 hours** of data. For higher-quality synthesis, we recommend using acoustic models like Soft-VC/Seed-VC to convert to a Mel spectrogram first, and then using the pre-trained vocoder like BigVGAN2.

### ToDo List
- [x] Add Seed-VC inference samples for comparison.
- [ ] Train the SoundStream Decoder using a larger dataset of high-quality audio. (I currently don't have the resources to train the model.)
