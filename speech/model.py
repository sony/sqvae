import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Encoder(nn.Module):
    def __init__(self, param_var_q, in_channels, channels, n_embeddings, embedding_dim, jitter=0.0):
        super(Encoder, self).__init__()
        self.param_var_q = param_var_q
        self.embedding_dim = embedding_dim

        if self.param_var_q == "gaussian_1":
            out_channels = embedding_dim
        elif self.param_var_q == "gaussian_3":
            out_channels = embedding_dim + 1
        elif self.param_var_q == "gaussian_4":
            out_channels = embedding_dim * 2
        else:
            raise Exception("Undefined param_var_q")

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, channels, 3, 1, 0, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, out_channels, 1)
        )

        log_var_q_scalar = torch.Tensor(1)
        log_var_q_scalar.fill_(10.0).log_()
        self.register_parameter("log_var_q_scalar", nn.Parameter(log_var_q_scalar))

        self.codebook = SQEmbedding(param_var_q, n_embeddings, embedding_dim)
        self.jitter = Jitter(jitter)

    def forward(self, mels, temperature):
        z = self.encoder(mels)
        z = z.transpose(1, 2)
        if self.param_var_q == "gaussian_1":
            log_var_q = self.log_var_q_scalar
        elif self.param_var_q == "gaussian_3" or self.param_var_q == "gaussian_4":
            log_var_q = z[:, :, self.embedding_dim:] + self.log_var_q_scalar
        else:
            raise Exception("Undefined param_var_q")
        z = z[:, :, :self.embedding_dim]
        z, loss, perplexity = self.codebook(z, log_var_q, temperature)
        z = self.jitter(z)
        return z, loss, perplexity

    def encode(self, mel):
        z = self.encoder(mel)
        z = z.transpose(1, 2)
        if self.param_var_q == "gaussian_1":
            log_var_q = self.log_var_q_scalar
        elif self.param_var_q == "gaussian_3" or self.param_var_q == "gaussian_4":
            log_var_q = z[:, :, self.embedding_dim:] + self.log_var_q_scalar
        else:
            raise Exception("Undefined param_var_q")
        z = z[:, :, :self.embedding_dim]
        z, indices = self.codebook.encode(z, log_var_q)
        return z, indices


class Jitter(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        prob = torch.Tensor([p / 2, 1 - p, p / 2])
        self.register_buffer("prob", prob)

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        else:
            batch_size, sample_size, channels = x.size()

            dist = Categorical(self.prob)
            index = dist.sample(torch.Size([batch_size, sample_size])) - 1
            index[:, 0].clamp_(0, 1)
            index[:, -1].clamp_(-1, 0)
            index += torch.arange(sample_size, device=x.device)

            x = torch.gather(x, 1, index.unsqueeze(-1).expand(-1, -1, channels))
        return x


class SQEmbedding(nn.Module):
    def __init__(self, param_var_q, n_embeddings, embedding_dim):
        super(SQEmbedding, self).__init__()
        self.param_var_q = param_var_q

        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.normal_()
        self.register_parameter("embedding", nn.Parameter(embedding))

    def encode(self, x, log_var_q):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)
        if self.param_var_q == "gaussian_1":
            log_var_q_flat = log_var_q.reshape(1, 1)
        elif self.param_var_q == "gaussian_3":
            log_var_q_flat = log_var_q.reshape(-1, 1)
        elif self.param_var_q == "gaussian_4":
            log_var_q_flat = log_var_q.reshape(-1, D)
        else:
            raise Exception("Undefined param_var_q")

        x_flat = x_flat.unsqueeze(2)
        log_var_flat = log_var_q_flat.unsqueeze(2)
        embedding = self.embedding.t().unsqueeze(0)
        precision_flat = torch.exp(-log_var_flat)
        distances = 0.5 * torch.sum(precision_flat * ((embedding - x_flat) ** 2), dim=1)

        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return quantized, indices

    def forward(self, x, log_var_q, temperature):
        M, D = self.embedding.size()
        batch_size, sample_size, channels = x.size()
        x_flat = x.reshape(-1, D)
        if self.param_var_q == "gaussian_1":
            log_var_q_flat = log_var_q.reshape(1, 1)
        elif self.param_var_q == "gaussian_3":
            log_var_q_flat = log_var_q.reshape(-1, 1)
        elif self.param_var_q == "gaussian_4":
            log_var_q_flat = log_var_q.reshape(-1, D)
        else:
            raise Exception("Undefined param_var_q")

        x_flat = x_flat.unsqueeze(2)
        log_var_flat = log_var_q_flat.unsqueeze(2)
        embedding = self.embedding.t().unsqueeze(0)
        precision_flat = torch.exp(-log_var_flat)
        distances = 0.5 * torch.sum(precision_flat * (embedding - x_flat) ** 2, dim=1)

        indices = torch.argmin(distances.float(), dim=-1)

        logits = -distances

        encodings = self._gumbel_softmax(logits, tau=temperature, dim=-1)
        quantized = torch.matmul(encodings, self.embedding)
        quantized = quantized.view_as(x)

        logits = logits.view(batch_size, sample_size, M)
        probabilities = torch.softmax(logits, dim=-1)
        log_probabilities = torch.log_softmax(logits, dim=-1)

        precision = torch.exp(-log_var_q)
        loss = torch.mean(0.5 * torch.sum(precision * (x - quantized) ** 2, dim=(1, 2))
                          + torch.sum(probabilities * log_probabilities, dim=(1, 2)))

        encodings = F.one_hot(indices, M).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, dim=-1):
        eps = torch.finfo(logits.dtype).eps
        gumbels = (
            -((-(torch.rand_like(logits).clamp(min=eps, max=1 - eps).log())).log())
        )  # ~Gumbel(0,1)
        gumbels_new = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels_new.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparameterization trick.
            ret = y_soft

        return ret


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, n_speakers,
                 speaker_embedding_dim, conditioning_channels,
                 fc_channels):
        super().__init__()
        self.speaker_embedding = nn.Embedding(n_speakers, speaker_embedding_dim)
        self.rnn = nn.GRU(in_channels + speaker_embedding_dim, conditioning_channels,
                          num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(fc_channels, out_channels)

    def forward(self, z, speakers):
        z = F.interpolate(z.transpose(1, 2), scale_factor=2)
        z = z.transpose(1, 2)

        speakers = self.speaker_embedding(speakers)
        speakers = speakers.unsqueeze(1).expand(-1, z.size(1), -1)

        z = torch.cat((z, speakers), dim=-1)
        z, _ = self.rnn(z)

        x = self.fc(z)
        return x

    def generate(self, z, speaker):
        output = self.forward(z, speaker)
        return output
