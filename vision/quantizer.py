import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical


def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape, device="cuda")
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    g = sample_gumbel(logits.size())
    y = logits + g
    return F.softmax(y / temperature, dim=-1)


def calc_distance(z_continuous, codebook, dim_dict):
    z_continuous_flat = z_continuous.view(-1, dim_dict)
    distances = (torch.sum(z_continuous_flat**2, dim=1, keepdim=True) 
                + torch.sum(codebook**2, dim=1)
                - 2 * torch.matmul(z_continuous_flat, codebook.t()))

    return distances



class VectorQuantizer(nn.Module):
    def __init__(self, size_dict, dim_dict, temperature=0.5):
        super(VectorQuantizer, self).__init__()
        self.size_dict = size_dict
        self.dim_dict = dim_dict
        self.temperature = temperature
    
    def forward(self, z_from_encoder, param_q, codebook, flg_train, flg_quant_det=False):
        return self._quantize(z_from_encoder, param_q, codebook,
                                flg_train=flg_train, flg_quant_det=flg_quant_det)
    
    def _quantize(self):
        raise NotImplementedError()
    
    def set_temperature(self, value):
        self.temperature = value
    
    def _calc_distance_bw_enc_codes(self):
        raise NotImplementedError()
    
    def _calc_distance_bw_enc_dec(self):
        raise NotImplementedError()


class GaussianVectorQuantizer(VectorQuantizer):
    def __init__(self, size_dict, dim_dict, temperature=0.5, param_var_q="gaussian_1"):
        super(GaussianVectorQuantizer, self).__init__(size_dict, dim_dict, temperature)
        self.param_var_q = param_var_q
    
    def _quantize(self, z_from_encoder, var_q, codebook, flg_train=True, flg_quant_det=False):
        bs, dim_z, width, height = z_from_encoder.shape
        z_from_encoder_permuted = z_from_encoder.permute(0, 2, 3, 1).contiguous()
        precision_q = 1. / torch.clamp(var_q, min=1e-10)

        logit = -self._calc_distance_bw_enc_codes(z_from_encoder_permuted, codebook, 0.5 * precision_q)
        probabilities = torch.softmax(logit, dim=-1)
        log_probabilities = torch.log_softmax(logit, dim=-1)
        
        # Quantization
        if flg_train:
            encodings = gumbel_softmax_sample(logit, self.temperature)
            z_quantized = torch.mm(encodings, codebook).view(bs, width, height, dim_z)
            avg_probs = torch.mean(probabilities.detach(), dim=0)
        else:
            if flg_quant_det:
                indices = torch.argmax(logit, dim=1).unsqueeze(1)
                encodings_hard = torch.zeros(indices.shape[0], self.size_dict, device="cuda")
                encodings_hard.scatter_(1, indices, 1)
                avg_probs = torch.mean(encodings_hard, dim=0)
            else:
                dist = Categorical(probabilities)
                indices = dist.sample().view(bs, width, height)
                encodings_hard = F.one_hot(indices, num_classes=self.size_dict).type_as(codebook)
                avg_probs = torch.mean(probabilities, dim=0)
            z_quantized = torch.matmul(encodings_hard, codebook).view(bs, width, height, dim_z)
        z_to_decoder = z_quantized.permute(0, 3, 1, 2).contiguous()
        
        # Latent loss
        kld_discrete = torch.sum(probabilities * log_probabilities, dim=(0,1)) / bs
        kld_continuous = self._calc_distance_bw_enc_dec(z_from_encoder, z_to_decoder, 0.5 * precision_q).mean()
        loss = kld_discrete + kld_continuous
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7)))

        return z_to_decoder, loss, perplexity

    def _calc_distance_bw_enc_codes(self, z_from_encoder, codebook, weight):        
        if self.param_var_q == "gaussian_1":
            distances = weight * calc_distance(z_from_encoder, codebook, self.dim_dict)
        elif self.param_var_q == "gaussian_2":
            weight = weight.tile(1, 1, 8, 8).view(-1,1)
            distances = weight * calc_distance(z_from_encoder, codebook, self.dim_dict)
        elif self.param_var_q == "gaussian_3":
            weight = weight.view(-1,1)
            distances = weight * calc_distance(z_from_encoder, codebook, self.dim_dict)
        elif self.param_var_q == "gaussian_4":
            z_from_encoder_flat = z_from_encoder.view(-1, self.dim_dict).unsqueeze(2)
            codebook = codebook.t().unsqueeze(0)
            weight = weight.permute(0, 2, 3, 1).contiguous().view(-1, self.dim_dict).unsqueeze(2)
            distances = torch.sum(weight * ((z_from_encoder_flat - codebook) ** 2), dim=1)

        return distances
        
    def _calc_distance_bw_enc_dec(self, x1, x2, weight):
        return torch.sum((x1-x2)**2 * weight, dim=(1,2,3))
    


class VmfVectorQuantizer(VectorQuantizer):
    def __init__(self, size_dict, dim_dict, temperature=0.5):
        super(VmfVectorQuantizer, self).__init__(size_dict, dim_dict, temperature)
    
    def _quantize(self, z_from_encoder, kappa_q, codebook, flg_train=True, flg_quant_det=False):
        bs, dim_z, width, height = z_from_encoder.shape
        z_from_encoder_permuted = z_from_encoder.permute(0, 2, 3, 1).contiguous()
        codebook_norm = F.normalize(codebook, p=2.0, dim=1)

        logit = -self._calc_distance_bw_enc_codes(z_from_encoder_permuted, codebook_norm, kappa_q)
        probabilities = torch.softmax(logit, dim=-1)
        log_probabilities = torch.log_softmax(logit, dim=-1)
        
        # Quantization
        if flg_train:
            encodings = gumbel_softmax_sample(logit, self.temperature)
            z_quantized = torch.mm(encodings, codebook_norm).view(bs, width, height, dim_z)
            avg_probs = torch.mean(probabilities.detach(), dim=0)
        else:
            if flg_quant_det:
                indices = torch.argmax(logit, dim=1).unsqueeze(1)
                encodings_hard = torch.zeros(indices.shape[0], self.size_dict, device="cuda")
                encodings_hard.scatter_(1, indices, 1)
                avg_probs = torch.mean(encodings_hard, dim=0)
            else:
                dist = Categorical(probabilities)
                indices = dist.sample().view(bs, width, height)
                encodings_hard = F.one_hot(indices, num_classes=self.size_dict).type_as(codebook)
                avg_probs = torch.mean(probabilities, dim=0)
            z_quantized = torch.matmul(encodings_hard, codebook_norm).view(bs, width, height, dim_z)
        z_to_decoder = z_quantized.permute(0, 3, 1, 2).contiguous()

        # Latent loss
        kld_discrete = torch.sum(probabilities * log_probabilities, dim=(0,1)) / bs
        kld_continuous = self._calc_distance_bw_enc_dec(z_from_encoder, z_to_decoder, kappa_q).mean()        
        loss = kld_discrete + kld_continuous
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7)))

        return z_to_decoder, loss, perplexity
 
    def _calc_distance_bw_enc_codes(self, z_from_encoder, codebook, kappa_q):
        z_from_encoder_flat = z_from_encoder.view(-1, self.dim_dict)
        distances = -kappa_q * torch.matmul(z_from_encoder_flat, codebook.t())

        return distances
    
    def _calc_distance_bw_enc_dec(self, x1, x2, weight):
        return torch.sum(x1 * (x1-x2) * weight, dim=(1,2,3))
