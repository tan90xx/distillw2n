from u2ss2u import StreamableModel
import torch
import torchaudio
import nemo.collections.asr as nemo_asr
DEVICE="cuda:0"

model = StreamableModel(
    batch_size=42,
    sample_rate=16_000,
    segment_length=32270,
    padding='same',
    dataset='timit')

checkpoint_path = './experiments/s2uu2s/epoch=440-step=409942.ckpt'
checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=True)
model.load_state_dict(checkpoint['state_dict'], strict=False)
model = model.to(DEVICE)
model.eval()

hubert_soft = torch.hub.load("bshall/hubert:main", f"hubert_soft").to(DEVICE)

speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
speaker_model = speaker_model.to(DEVICE)
speaker_model.eval()

x_trg, sr = torchaudio.load(f'./raw/gt/s000u003n.wav')
spkemb = speaker_model.infer_segment(x_trg.squeeze(0))[0]

x, sr = torchaudio.load('./raw/gt/s000u003w.wav')
x = torchaudio.functional.resample(x, sr, 16000)
# z, hubert = model(x.to(DEVICE), spkemb.to(DEVICE))
# torchaudio.save('test0.wav', z.squeeze(1).detach().cpu(), 16000)

# spec = model.spec.to(DEVICE)
# encoder = model.spec.to(DEVICE)
reencoder = model.reencoder.to(DEVICE)
decoder = model.decoder.to(DEVICE)

hubert_soft = torch.hub.load("bshall/hubert:main", f"hubert_soft").to(DEVICE)
hubert = hubert_soft.units(x.unsqueeze(0).to(DEVICE))
hubert = hubert.clone().to(DEVICE)
hubert = torch.transpose(hubert, -1, -2)
z = reencoder(hubert.to(DEVICE), spkemb.to(DEVICE))
z = decoder(z.to(DEVICE))
torchaudio.save('test1.wav', z.squeeze(1).detach().cpu(), 16000)