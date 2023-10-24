<div align="center">

# SynthNet Transfer Learning

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) <br>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

[![Paper](https://img.shields.io/badge/%20paper-blue?style=flat)](https://arxiv.org/pdf/2310.04757.pdf)
<!-- [![Conference](img)](link) -->

</div>

## Description

SynthNet Transfer Learning is a transfer learning framework based on Pytorch Lightning, Omegaconf and Hydra.

## Installation

```bash
# clone project
git clone git@gitlab.bht-berlin.de:iisy/SynthNet/synthnet-transfer-learning.git
cd synthnet-transfer-learning

# [OPTIONAL] create virtual environment and activate it
pyenv virtualenv 3.10.4 synthnet-transfer-learning
pyenv shell synthnet-transfer-learning

# install requirements
pip install -r requirements.txt
```

## Dataset Preparation

### Prepare VisDa-2017 Image Classification dataset for experiments

- Get the Visda2017 Dataset from [the official website](http://ai.bu.edu/visda-2017/) and store it under `data/visda2017`.
  Ensure the following directory structure: `data/visda2017/<split>/<class>/<images>`. You also have to rename and restructure the test set using the [released labels](https://raw.githubusercontent.com/VisionLearningGroup/taskcv-2017-public/master/classification/data/image_list.txt) for image ids.

### Prepare Topex-Printer dataset for experiments

- Download the dataset [HERE](https://huggingface.co/datasets/ritterdennis/topex-printer/resolve/main/topex-printer.zip)
- move dataset directory to `data/topex-printer`

### Prepare other datasets

- Ensure directory structure: `data/<my_dataset>/<split>/<class>/<images>` (or change/add datamodules to support your preferred dataset structure)

### Notes

- Logging is setup for [Weights & Biases](https://wandb.com)
- We integrated [tllib](https://github.com/thuml/Transfer-Learning-Library) directly into the project and fixed some issues for usage with pytorch-lightning 2.0

## How To Run

### Test your setup

Test run using reduced dataset size to check if setup works

```bash
python src/train.py -m 'experiment=STL-visda2017/src_only/vitb16_ch_sgd1e-3' data.toy=True logger.wandb.project=STL-test
```

### Best performing VisDa-2017 experiment setup

First, tune the classification head by running

```bash
python src/train.py -m 'experiment=STL-visda2017/src_only/swinv2_ch_sgd1e-3'
```

Then navigate to the best saved checkpoint from this run and initialize the model weights from it to continue training in an Unsupervised Domain Adaptation (UDA) setup:

```bash
python src/train.py -m 'experiment=STL-visda2017/uda/swinv2_ch-uda_cdan-mcc_adamw1e-5_warmupcalr_augmix' model.fine_tuning_checkpoint=<PATH/TO/MY/CHECKPOINT/epoch_0XX.ckpt>
# The checkpoint path looks like this: "out/synthnet-transfer-learning-outputs/train/multiruns/STL-visda2017/swinv2_ch_sgd1e-3/2023-04-21_14-13-45/5/checkpoints/epoch_000.ckpt"
```

### Best performing Topex-Printer experiment setup

First, tune the classification head by running

```bash
python src/train.py -m 'experiment=STL-topex/src_only/swinv2_ch_sgd1e-3'
```

Then navigate to the best saved checkpoint from this run and initialize the model weights from it to continue training in an Unsupervised Domain Adaptation (UDA) setup:

```bash
python src/train.py -m 'experiment=STL-topex/uda/swinv2_ch-uda_cdan-mcc_adamw1e-5_warmupcalr_augmix' model.fine_tuning_checkpoint=<PATH/TO/MY/CHECKPOINT/epoch_0XX.ckpt>
# The checkpoint path looks like this: "out/synthnet-transfer-learning-outputs/train/multiruns/STL-topex/src_only/swinv2_ch_sgd1e-3.yaml/2023-06-09_10-41-30/2/checkpoints/epoch_019.ckpt"
```

### General

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line without changing any config files. See the example below to change trainer.max_epochs to 10 and data.batch_size to 64.

```bash
python src/train.py experiment=my_experiment trainer.max_epochs=10 data.batch_size=64
```
## Citation
Please cite our [paper](https://arxiv.org/abs/2310.04757) if you use this Code or the Topex-Printer dataset. 
```
@misc{ritter2023cad,
      title={CAD Models to Real-World Images: A Practical Approach to Unsupervised Domain Adaptation in Industrial Object Classification}, 
      author={Dennis Ritter and Mike Hemberger and Marc Hönig and Volker Stopp and Erik Rodner and Kristian Hildebrand},
      year={2023},
      eprint={2310.04757},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
