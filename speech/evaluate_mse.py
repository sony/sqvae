import hydra
import hydra.utils as utils

import json
from pathlib import Path
import torch
import numpy as np
import librosa
from tqdm import tqdm

from preprocess import preemphasis
from model import Encoder, Decoder


@hydra.main(config_path="config/mse_evaluation.yaml")
def evaluate_mse(cfg):
    dataset_path = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path
    with open(dataset_path / "speakers.json") as file:
        speakers = sorted(json.load(file))

    evaluation_list_path = Path(utils.to_absolute_path(cfg.evaluation_list))
    with open(evaluation_list_path) as file:
        evaluation_list = json.load(file)

    in_dir = Path(utils.to_absolute_path(cfg.in_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(**cfg.model.encoder)
    decoder = Decoder(**cfg.model.decoder)
    encoder.to(device)
    decoder.to(device)

    print("Load checkpoint from: {}:".format(cfg.checkpoint))
    checkpoint_path = utils.to_absolute_path(cfg.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder.eval()
    decoder.eval()

    mse = []
    length = []
    for wav_path, speaker_id in tqdm(evaluation_list):
        wav_path = in_dir / wav_path
        wav, _ = librosa.load(
            wav_path.with_suffix(".wav"),
            sr=cfg.preprocessing.sr)
        wav = wav / np.abs(wav).max() * 0.999

        mel = librosa.feature.melspectrogram(
            preemphasis(wav, cfg.preprocessing.preemph),
            sr=cfg.preprocessing.sr,
            n_fft=cfg.preprocessing.n_fft,
            n_mels=cfg.preprocessing.n_mels,
            hop_length=cfg.preprocessing.hop_length,
            win_length=cfg.preprocessing.win_length,
            fmin=cfg.preprocessing.fmin,
            power=1)
        logmel = librosa.amplitude_to_db(mel, top_db=cfg.preprocessing.top_db)
        logmel = logmel / cfg.preprocessing.top_db + 1

        if logmel.shape[1] % 2 == 1:
            logmel = logmel[:, :-1]

        mel = torch.FloatTensor(logmel).unsqueeze(0).to(device)
        speaker = torch.LongTensor([speakers.index(speaker_id)]).to(device)
        with torch.no_grad():
            z, _ = encoder.encode(mel)
            output = decoder.generate(z, speaker)

        mse.append(torch.sum((output[0].transpose(0, 1) - mel[0, :, 1:-1]) ** 2).item())
        length.append(mel.size(2) - 2)

    mse = sum(mse) * (cfg.preprocessing.top_db ** 2) / (cfg.preprocessing.n_mels * sum(length))
    print("MSE: {}".format(mse))


if __name__ == "__main__":
    evaluate_mse()
