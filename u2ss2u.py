# (c) 2024-2025 Tan Tianyi
# This code is adopted from an unofficial SoundStream implementation in Pytorch.
# The original implementation can be found at https://github.com/kaiidams/soundstream-pytorch.
# We are using it under the MIT license. Thanks to the original author for providing this great work.

from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
try:
    import pytorch_lightning as pl
except ImportError:
    class pl:
        class LightningModule:
            pass
        class Callback:
            pass

from datahelper import TIMIT, WTIMIT, LJSPEECH, LIBRITTS, WHISPER
from models.s2u import call_feature_by_name, DVAEDecoder
from models.u2s import Reencoder, Decoder
from models.discriminators import WaveDiscriminator, ReconstructionLoss, STFTDiscriminator
from models.loss import t_axis_distill_loss, MultiScaleMelSpectrogramLoss
from utils.config import Config
from utils.audioprep import process_signal
from utils.s2f0 import load_F0_models, wav2F0
from pesq import pesq
import nemo.collections.asr as nemo_asr
     
    
class StreamableModel(pl.LightningModule):
    def __init__(
        self,
        n_channels: int = 16,
        padding: str = "same",
        n_reencoder_layer: int = 1,
        n_encoder_layer: int = 12,
        batch_size: int = 32,
        n_embed_dim: int = 256,
        sample_rate: int = 16_000,
        n_mels: int = 80,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 320,
        segment_length: int = 32270,
        lr: float = 1e-6,
        b1: float = 0.5,
        b2: float = 0.9,
        dataset: str = 'ljspeech',
        reen_nn_type: str = 'adapt',
        feature_type: str = 'mfcc',
        trainable: bool = True,
        pseudo_rate: float = 0.4,
        datasets_root: str = 'YOURPATH',
        F0_model_path: str = './libs/JDC/bst.t7',

    ) -> None:
        # https://arxiv.org/pdf/2009.02095.pdf
        # 2. Method
        # SEANet uses Adam with lr=1e-4, beta1=0.5, beta2=0.9
        # batch_size=16
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.spec = call_feature_by_name(feature_type, trainable)
        self.reencoder = Reencoder(n_layers=n_reencoder_layer, wavenet_embed_dim=n_embed_dim, nn_type=reen_nn_type)
        self.encoder = DVAEDecoder(idim=n_embed_dim, odim=n_embed_dim, n_layer=n_encoder_layer)
        self.decoder = Decoder(n_channels=n_channels, padding=padding)
        # self.linear = nn.Linear(256, 512)

        self.speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large", map_location=torch.cuda.current_device())
        self.speaker_model.eval()

        self.wave_discriminators = nn.ModuleList([
            WaveDiscriminator(resolution=1),
            WaveDiscriminator(resolution=2),
            WaveDiscriminator(resolution=4)
        ])
        self.rec_loss = ReconstructionLoss()
        self.stft_discriminator = STFTDiscriminator()

        self.to_mel = torchaudio.transforms.MelSpectrogram(n_mels=n_mels, sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        self.hubert_loss = t_axis_distill_loss()
        self.energy_loss = MultiScaleMelSpectrogramLoss(sampling_rate=sample_rate)

        self.hubert_soft = torch.hub.load("bshall/hubert:main", f"hubert_soft").to(torch.cuda.current_device())
        self.pitch_extractor = load_F0_models(F0_model_path, device="cuda:{}".format(torch.cuda.current_device()))
        self.segment_length = segment_length
        self.datasets_root = datasets_root
        self.pseudo_rate = pseudo_rate

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        optimizer_g = torch.optim.Adam(
            chain(
                self.spec.parameters(),
                self.encoder.parameters(),
                self.reencoder.parameters(),
                self.decoder.parameters(),
            ),
            lr=lr, betas=(b1, b2))
        optimizer_d = torch.optim.Adam(
            chain(
                self.wave_discriminators.parameters(),
                self.stft_discriminator.parameters()
            ),
            lr=lr, betas=(b1, b2))
        # scheduler_d = torch.optim.lr_scheduler.StepLR(
        #     optimizer_d, step_size=2, gamma=0.95
        # )
        return [optimizer_g, optimizer_d], []

    def forward(self, input, spkemb):
        spectrogram = self.spec(input)
        spectrogram = spectrogram.transpose(-1, -2)
        x = self.encoder(spectrogram)
        hubert_like = torch.nn.functional.pad(x, (0, 0, 0, 1, 0, 0))
        x = torch.transpose(hubert_like, -1, -2)
        # hubert = self.hubert_soft.units(input)
        # hubert = torch.nn.functional.pad(hubert, (0, 0, 0, 2, 0, 0))
        # x = torch.transpose(hubert, -1, -2)
        x = self.reencoder(x, spkemb)
        x = self.decoder(x)
        return x, hubert_like

    def training_step(self, batch, batch_idx):
        optimizer_g, optimizer_d = self.optimizers()
        # sch = self.lr_schedulers() 
        inputs = batch[:, None, :] # 1:normal 2:ppw 3:vad
        input = inputs[:, :,self.segment_length*2:self.segment_length*3]
        input_0 = inputs[:, :,:32270]
        # if random.random() < self.pseudo_rate:
        #     input_0 = inputs[:, :, self.segment_length*1:self.segment_length*2] # normal
        # else:
        #     input_0 = inputs[:, :, :self.segment_length]  # ppw
        spkemb = torch.cat([self.speaker_model.infer_segment(w16)[0] for w16 in input.squeeze().squeeze().cpu()], dim=0)

        # train generator
        self.toggle_optimizer(optimizer_g)
        output, hubert_like = self.forward(input_0, spkemb)

        # F0 Loss
        to_mel = self.to_mel.to(input.device)
        mels = to_mel(input).squeeze()
        pred_mels = to_mel(output).squeeze()
        mel_mean, mel_std = -4, 4
        mels = (torch.log(1e-5 + mels) - mel_mean) / mel_std
        pred_mels = (torch.log(1e-5 + pred_mels) - mel_mean) / mel_std
        F0_real = wav2F0(mels, self.pitch_extractor, input.device, norm=False)
        F0_pred = wav2F0(pred_mels, self.pitch_extractor, input.device, norm=False)
        f0_loss = F.smooth_l1_loss(F0_real, F0_pred)
        self.log("f0_loss", f0_loss, prog_bar=False)
        # Energy Loss
        energy_loss = self.energy_loss(input, output)
        self.log("energy_loss", energy_loss, prog_bar=False)
        # Content Loss 
        pred_hubert = self.hubert_soft.units(output.to(torch.cuda.current_device()))
        pred_hubert = pred_hubert.to(input.device)
        content_loss = self.hubert_loss(pred_hubert, hubert_like)
        self.log("content_loss", content_loss, prog_bar=False)
        # Speaker Embedding Loss
        pred_spkemb = torch.cat([self.speaker_model.infer_segment(w16)[0] for w16 in output.squeeze().squeeze().cpu()], dim=0)
        spk_loss = self.hubert_loss(pred_spkemb, spkemb)
        self.log("spk_loss", spk_loss, prog_bar=False)

        stft_out = self.stft_discriminator(output)
        g_stft_loss = torch.mean(torch.relu(1 - stft_out))
        self.log("g_stft_loss", g_stft_loss)

        g_wave_loss = 0
        g_feat_loss = 0
        for i in range(3):
            feats1 = self.wave_discriminators[i](input)
            feats2 = self.wave_discriminators[i](output)
            assert len(feats1) == len(feats2)
            g_wave_loss += torch.mean(torch.relu(1 - feats2[-1]))
            g_feat_loss += sum(torch.mean(
                torch.abs(f1 - f2))
                for f1, f2 in zip(feats1[:-1], feats2[:-1])) / (len(feats1) - 1)
        self.log("g_wave_loss", g_wave_loss / 3)
        self.log("g_feat_loss", g_feat_loss / 3)

        g_rec_loss = self.rec_loss(output[:, 0, :], input[:, 0, :])
        self.log("g_rec_loss", g_rec_loss, prog_bar=True)

        g_feat_loss = g_feat_loss / 3
        g_adv_loss = (g_stft_loss + g_wave_loss) / 4
        g_loss = g_adv_loss  + g_rec_loss +  100 * g_feat_loss  + 0.5 * f0_loss  + 0.5 * energy_loss  + spk_loss + content_loss

        self.log("g_loss", g_loss, prog_bar=True)

        self.manual_backward(g_loss)
        # torch.nn.utils.clip_grad_norm_(self.spec.parameters(), max_norm=0.5)
        # torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=0.5)
        # torch.nn.utils.clip_grad_norm_(self.reencoder.parameters(), max_norm=0.5)
        # torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=0.5)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)
        # sch.step()

        # train discriminator
        output, hubert_like = self.forward(input, spkemb)

        stft_out = self.stft_discriminator(input)
        d_stft_loss = torch.mean(torch.relu(1 - stft_out))
        stft_out = self.stft_discriminator(output)
        d_stft_loss += torch.mean(torch.relu(1 + stft_out))

        d_wave_loss = 0
        for i in range(3):
            feats = self.wave_discriminators[i](input)
            d_wave_loss += torch.mean(torch.relu(1 - feats[-1]))
            feats = self.wave_discriminators[i](output)
            d_wave_loss += torch.mean(torch.relu(1 + feats[-1]))

        d_loss = (d_stft_loss + d_wave_loss) / 4

        self.log("d_stft_loss", d_stft_loss)
        self.log("d_wave_loss", d_wave_loss / 3)

        d_loss = (d_stft_loss + d_wave_loss) / 4
        self.log("d_loss", d_loss, prog_bar=True)

        self.manual_backward(d_loss)
        # if (batch_idx + 1) % 2 == 0:
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def validation_step(self, batch, batch_idx):
        inputs = batch[:, None, :]
        input = inputs[:, :, :32270] # normal / ppw
        whisper = inputs[:, :,32270:32270*2] # ppw
        spkemb = torch.cat([self.speaker_model.infer_segment(w16)[0] for w16 in input.squeeze().squeeze().cpu()], dim=0)
        with torch.no_grad():
            output, _ = self.forward(whisper, spkemb)
            val_pesq_tot = 0
            MAX_WAV_VALUE = 32767.0
            for y_16k, y_g_hat_16k in zip(input, output):
                y_int_16k = (y_16k[0] * MAX_WAV_VALUE).short().cpu().numpy().astype(int)
                y_g_hat_int_16k = (y_g_hat_16k[0] * MAX_WAV_VALUE).short().cpu().numpy().astype(int)
                val_pesq_tot += pesq(16000, y_int_16k, y_g_hat_int_16k, "wb")
        self.log("val_pesq", val_pesq_tot, on_epoch=True, prog_bar=True, sync_dist=True)

    def train_dataloader(self):
        return self._make_dataloader(True)
    
    def val_dataloader(self):
        return self._make_dataloader_val()

    def _make_dataloader(self, train: bool):
        
        class VoiceDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, sample_rate, segment_length):
                self._dataset = dataset
                self._sample_rate = sample_rate
                self._segment_length = segment_length

            def __getitem__(self, index):
                import random
                x, x_p, x_v, sample_rate, *_ = self._dataset[index]
                target_len = min(x.shape[-1], x_p.shape[-1], x_v.shape[-1])
                x = process_signal(x,sample_rate, self._sample_rate, target_len, self._segment_length)
                x_p = process_signal(x_p,sample_rate, self._sample_rate, target_len, self._segment_length)
                x_v = process_signal(x_v,sample_rate, self._sample_rate, target_len, self._segment_length)
                pos = random.randint(0, x.shape[0] - self._segment_length)
                x = x[pos:pos + self._segment_length]
                x_p = x_p[pos:pos + self._segment_length]
                x_v = x_v[pos:pos + self._segment_length]
                output = torch.cat((x, x_p, x_v), 0)
                return output

            def __len__(self):
                return len(self._dataset)
            
        def collate(examples):
            return torch.stack(examples)

        if self.hparams.dataset == 'ljspeech':
            ds = LJSPEECH(self.datasets_root)
        elif self.hparams.dataset == 'libritts':
            ds = LIBRITTS(self.datasets_root)
        elif self.hparams.dataset == 'timit':
            ds = TIMIT(self.datasets_root, training=True)
        elif self.hparams.dataset == 'wtimit':
            ds = WTIMIT(self.datasets_root)
        else:
            raise ValueError()
        ds = VoiceDataset(ds, self.hparams.sample_rate, self.hparams.segment_length)

        # subset_len = int(0.2 * len(ds))
        # ds = torch.utils.data.Subset(ds, range(subset_len))

        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.hparams['batch_size'], shuffle=True,
            collate_fn=collate, num_workers=7)
        return loader

    def _make_dataloader_val(self, sub_rate: float=0.1):

        class VoiceDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, sample_rate, segment_length):
                self._dataset = dataset
                self._sample_rate = sample_rate
                self._segment_length = segment_length

            def __getitem__(self, index):
                import random
                x, x_p, sample_rate, *_ = self._dataset[index]
                target_len = min(x.shape[-1], x_p.shape[-1])
                x = process_signal(x,sample_rate, self._sample_rate, target_len, self._segment_length)
                x_p = process_signal(x_p,sample_rate, self._sample_rate, target_len, self._segment_length)
                pos = random.randint(0, x.shape[0] - self._segment_length)
                x = x[pos:pos + self._segment_length]
                x_p = x_p[pos:pos + self._segment_length]
                output = torch.cat((x, x_p), 0)
                return output

            def __len__(self):
                return len(self._dataset)

        def collate(examples):
            return torch.stack(examples)

        ds = WHISPER(self.datasets_root)
        
        subset_len = int(sub_rate * len(ds))
        ds = torch.utils.data.Subset(ds, range(subset_len))

        ds = VoiceDataset(ds, self.hparams.sample_rate, self.hparams.segment_length)

        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.hparams['batch_size'], shuffle=False,
            collate_fn=collate, num_workers=7)
        return loader


def train():
    config = {
        # Configuration ID
        'id': "experiments",
        'name': "s2uu2s",
        'version': "s2uu2s-libri-ti",
        # Training configuration
        'batch_size': 32, #s2u 128, s2uu2s 32
        'save_checkpoint_dir': "",
        'restore_checkpoint_path': "./experiments/s2uu2s/epoch=440-step=409942.ckpt",
        'resume_training': True,
        'training_epochs': 10000,
        'log_every_n_steps': 2,
        'dataset': 'libritts',
        'feature_type': 'mfcc',
        'reen_nn_type': 'adapt',
    }
    config = Config(config)
    model = StreamableModel(
        n_channels=config.n_channels,
        n_embed_dim=config.n_embed_dim,
        n_encoder_layer=config.n_encoder_layer,
        padding=config.padding,
        batch_size=config.batch_size,
        sample_rate=config.sample_rate,
        segment_length=config.segment_length,
        lr=config.lr,
        b1=config.b1,
        b2=config.b2,
        dataset=config.dataset,
        feature_type=config.feature_type,
        reen_nn_type=config.reen_nn_type,
        pseudo_rate=config.pseudo_rate,
        datasets_root=config.datasets_root,
        F0_model_path=config.F0_model_path)
    pl.seed_everything(config.seed, workers=True)
    trainer = pl.Trainer(
        max_epochs=config.training_epochs,
        log_every_n_steps=config.log_every_n_steps,
        precision='16-mixed',
        logger=pl.loggers.TensorBoardLogger(config.save_checkpoint_dir+config.id, name=config.name, version=config.version),
        callbacks=[
            pl.callbacks.ModelCheckpoint(save_last=True, monitor='val_pesq', save_top_k=2, mode='max')
        ],
        strategy='ddp_find_unused_parameters_true'
    )
    trainer.fit(
        model,
        ckpt_path=config.restore_checkpoint_path
    )

    return model


if __name__ == "__main__":
    train()