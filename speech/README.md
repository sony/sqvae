# SQ-VAE (Speech Dataset)

This repository contains the official Pytorch implementation of SQ-VAE.

## Quick Start

### Requirements

1.  Ensure you have Python 3 and PyTorch 1.4 or greater.

2.  Install [NVIDIA/apex](https://github.com/NVIDIA/apex) for mixed precision training.

3.  Install pip dependencies:
    ```
    pip install -r requirements.txt
    ```

### Data and Preprocessing

1.  Download and extract the [ZeroSpeech2020 dataset](https://download.zerospeech.com/).
    For reproduction, only `zerospeech2020.z01`, `zerospeech2020.z02`, and `zerospeech2020.zip` are required
    (follow the official instruction to extract the dataset).

2.  Download the train/test splits [here](https://github.com/bshall/ZeroSpeech/releases/tag/v0.1) 
    and extract in the root directory of the repo.
    
3.  Preprocess audio and extract train/test log-Mel spectrograms:
    ```
    python preprocess.py in_dir=/path/to/dataset dataset=2019/english
    ```
    Note: `in_dir` must be the path to the `2019` folder. 
    ```
    e.g. python preprocess.py in_dir=../datasets/2020/2019 dataset=2019/english
    ```
    
### Training
   
Train a model:
```
python train.py checkpoint_dir=path/to/checkpoint_dir dataset=2019/english
```
```
e.g. python train.py checkpoint_dir=checkpoints/2019english dataset=2019/english
```
Note: The default parameterization of the variance is Gaussian SQ-VAE (IV) `"gaussian_4"`.
You can switch the parameterizations in `config/model/default.yaml`:
Gaussian SQ-VAE (I) `"gaussian_1"`, Gaussian SQ-VAE (III) `"gaussian_3"`, and Gaussian SQ-VAE (IV) `"gaussian_4"`.

### Evaluation
    
#### Mean Squared Error

```
python evaluate_mse.py checkpoint=path/to/checkpoint in_dir=path/to/wavs evaluation_list=path/to/evaluation_list dataset=2019/english
```
Note: the `evaluation list` is a `json` file:
```
[
    [
        "english/train/parallel/voice/V001_0000000047.wav",
        "V001"
    ]
]
```
containing a list of items with a) the path (relative to `in_dir`) of the source `wav` files;
and b) the target speaker (see `datasets/2019/english/speakers.json` for a list of options).
```
e.g. python evaluate_mse.py checkpoint=checkpoints/2019english/model.ckpt-500000.pt in_dir=../datasets/2020/2019 evaluation_list=datasets/2019/english/mse_evaluation.json dataset=2019/english
```
    
#### ABX Score
    
1.  Install [bootphon/zerospeech2020](https://github.com/bootphon/zerospeech2020).

2.  Encode test data for evaluation:
    ```
    python encode.py checkpoint=path/to/checkpoint out_dir=path/to/out_dir dataset=2019/english
    ```
    ```
    e.g. python encode.py checkpoint=checkpoints/2019english/model.ckpt-500000.pt out_dir=submission/2019/english/test dataset=2019/english
    ```
    
3. Run ABX evaluation script (see [bootphon/zerospeech2020](https://github.com/bootphon/zerospeech2020)).

## References

This work is based on:

1.  Niekerk, Nortje, and Kamper. ["Vector-quantized neural networks for acoustic unit discovery in the ZeroSpeech 2020 challenge."](https://arxiv.org/abs/2005.09409)
    INTERSPEECH. 2020.

2.  Chorowski, Jan, et al. ["Unsupervised speech representation learning using wavenet autoencoders."](https://arxiv.org/abs/1901.08810)
    IEEE/ACM transactions on audio, speech, and language processing 27.12 (2019): 2041-2053.

3.  Lorenzo-Trueba, Jaime, et al. ["Towards achieving robust universal neural vocoding."](https://arxiv.org/abs/1811.06292)
    INTERSPEECH. 2019.
    
4.  van den Oord, Aaron, and Oriol Vinyals. ["Neural discrete representation learning."](https://arxiv.org/abs/1711.00937)
    Advances in Neural Information Processing Systems. 2017.

## Citation

```
@INPROCEEDINGS{takida2022sq-vae,
    author={Takida, Yuhta and Shibuya, Takashi and Liao, WeiHsiang and Lai, Chieh-Hsin and Ohmura, Junki and Uesaka, Toshimitsu and Murata, Naoki and Takahashi Shusuke and Kumakura, Toshiyuki and Mitsufuji, Yuki},
    title={SQ-VAE: Variational Bayes on Discrete Representation with Self-annealed Stochastic Quantization},
    booktitle={International Conference on Machine Learning},
    year={2022},
    }
```
