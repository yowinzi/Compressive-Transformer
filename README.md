# Compressive_Transformer

## Introduction

This is an implementation of Compressive Transformer. This project is about doing ASR(Automatic Speech Recognition), NMT(Neural Machine Translation) by Compressive Transformer Architecture.

## Requirements
- ESPnet
- Python 3.7


## Setup

Because this project is implemented by ESPNET, you can also chenck the ESPNET documentaiton. Using insturction shown as below to set ESPNET environment.

```sh
cd tools
make
```


## Training

```sh
cd egs/aishell/com_asr
./run_com.sh --stage -1  
```

```sh
cd egs/aishell/com_asr
./run_com.sh --stage 4 
```

## Testing
```sh
cd egs/aishell/com_asr
./run_com.sh --stage 5 --recog_set "dev test" --tag "Path of Model"
```

## Reference
[Compressive Transformers for Long-Range Sequence Modelling](https://arxiv.org/pdf/1911.05507.pdf)

[ESPNET](https://github.com/espnet/espnet)

