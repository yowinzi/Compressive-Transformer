# Compressive_Transformer

# Introduction
This Project is an implementation about Compress Transformer. If you want to know the detail of Compressive Transformer, please check the reference.

## Requirements
- ESPnet
- Python 3.7

## Setup

You can also download the original package and use my code to overwrite it. Or just download my code and run the instruction shown as below.

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
./run_com.sh --stage 5 --recog_set "dev test" --tag "Path of Model"
```

## Reference
[ESPNET](https://github.com/espnet/espnet)

[Compressive Transformers for Long-Range Sequence Modelling](https://arxiv.org/pdf/1911.05507.pdf)



