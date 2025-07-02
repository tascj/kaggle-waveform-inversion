# Yale/UNC-CH - Geophysical Waveform Inversion

[Competition](https://www.kaggle.com/competitions/waveform-inversion)

## Requirements

### Hardware

A100 SXM 80G x4

### Software

Base Image
```
nvcr.io/nvidia/pytorch:25.05-py3
```

Packages
```
timm
detectron2  # set CUDA_VISIBLE_DEVICES="" to skip building cuda extensions
```

## Generate Data

This will take roughly 13TB of disk space.
```
python scripts/dump_dataset.py
. generate_data.sh
```


## Training

A copy of annotation files is placed in `../data/`. Copy to `../artifacts/`.

```
torchrun --nproc_per_node=4 main.py configs/release02.py
python scripts/convert_18to35.py
torchrun --nproc_per_node=4 main.py configs/release03.py
torchrun --nproc_per_node=4 main.py configs/release04.py
```


## Inference

```
python predict.py configs/release02.py --load-from ../artifacts/release02/epoch_10.pth --input-root ../data/test --output-root ../submissions/release02/
python predict.py configs/release03.py --load-from ../artifacts/release03/epoch_1.pth --input-root ../data/test --output-root ../submissions/release03/
python predict.py configs/release04.py --load-from ../artifacts/release04/epoch_1.pth --input-root ../data/test --output-root ../submissions/release04/
```
