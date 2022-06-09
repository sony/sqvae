import hydra
from hydra import utils
from itertools import chain
from pathlib import Path
from tqdm import tqdm
import math

import apex.amp as amp
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import SpeechDataset
from model import Encoder, Decoder


def mse_loss_arelbo(input, target):
    # "Preventing Posterior Collapse Induced by Oversmoothing in Gaussian VAE"
    # https://arxiv.org/abs/2102.08663
    return 0.5 * (target.numel() // target.size(0)) * torch.log(torch.mean((input - target) ** 2))


def save_checkpoint(encoder, decoder, optimizer, amp, scheduler, step, checkpoint_dir):
    checkpoint_state = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "amp": amp.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step}
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / "model.ckpt-{}.pt".format(step)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path.stem))


@hydra.main(config_path="config/train.yaml")
def train_model(cfg):
    tensorboard_path = Path(utils.to_absolute_path("tensorboard")) / cfg.checkpoint_dir
    checkpoint_dir = Path(utils.to_absolute_path(cfg.checkpoint_dir))
    writer = SummaryWriter(tensorboard_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(**cfg.model.encoder)
    decoder = Decoder(**cfg.model.decoder)
    encoder.to(device)
    decoder.to(device)

    optimizer = optim.Adam(
        chain(encoder.parameters(), decoder.parameters()),
        lr=cfg.training.optimizer.lr)
    [encoder, decoder], optimizer = amp.initialize([encoder, decoder], optimizer, opt_level="O1")
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.training.scheduler.milestones,
        gamma=cfg.training.scheduler.gamma)

    if cfg.resume:
        print("Resume checkpoint from: {}:".format(cfg.resume))
        resume_path = utils.to_absolute_path(cfg.resume)
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        amp.load_state_dict(checkpoint["amp"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = checkpoint["step"]
    else:
        global_step = 0

    root_path = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path
    dataset = SpeechDataset(
        root=root_path,
        hop_length=cfg.preprocessing.hop_length,
        sr=cfg.preprocessing.sr,
        sample_frames=cfg.training.sample_frames)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.n_workers,
        pin_memory=True,
        drop_last=True)

    n_epochs = cfg.training.n_steps // len(dataloader) + 1
    start_epoch = global_step // len(dataloader) + 1

    for epoch in range(start_epoch, n_epochs + 1):
        average_recon_loss = average_kl = average_perplexity = 0

        for i, (mels, speakers) in enumerate(tqdm(dataloader), 1):
            mels, speakers = mels.to(device), speakers.to(device)

            optimizer.zero_grad()

            temperature = cfg.training.temperature.init * math.exp(-cfg.training.temperature.decay * global_step)
            z, kl, perplexity = encoder(mels, temperature)
            output = decoder(z, speakers)
            recon_loss = mse_loss_arelbo(output.transpose(1, 2), mels[:, :, 1:-1])
            loss = recon_loss + kl

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
            optimizer.step()
            scheduler.step()

            average_recon_loss += (recon_loss.item() - average_recon_loss) / i
            average_kl += (kl.item() - average_kl) / i
            average_perplexity += (perplexity.item() - average_perplexity) / i

            global_step += 1

            if global_step % cfg.training.checkpoint_interval == 0:
                save_checkpoint(
                    encoder, decoder, optimizer, amp,
                    scheduler, global_step, checkpoint_dir)

        writer.add_scalar("recon_loss/train", average_recon_loss, global_step)
        writer.add_scalar("kl/train", average_kl, global_step)
        writer.add_scalar("average_perplexity", average_perplexity, global_step)

        print("epoch:{}, recon loss:{:.2E}, kl:{:.2E}, perplexity:{:.3f}"
              .format(epoch, average_recon_loss, average_kl, average_perplexity))


if __name__ == "__main__":
    train_model()
