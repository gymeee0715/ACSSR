# Learning Adapters for Code-Switching Speech Recognition
This repository is developed under huggingface Framework

### Requirements

- datasets >= 1.18.0
- torch >= 1.5
- torchaudio
- librosa
- jiwer
- evaluate
- numpy
- pandas
- jieba
- editdistance
- tensorboard
- fairscale
- seaborn
- accelerate
- spacy

## Installatation
1. Install huggingface
```
> cd transformers
> pip install -e .
```


## dataset  and pretrained weight

* dataset
```
W:\Chun-Yi_He\ASR_data\NTUT\dataset_NTUT
W:\Chun-Yi_He\ASR_data\ASCEND
```
* pretrained weight
```
W:\Chun-Yi_He\pretrained_weight
```
## FIle structure
```
|_ /ASCEND/ 
    |_ dataset_NTUT (NTUT AB01 dataset)  
    |_ waves (ASCEND dataset)
```

## Model training 
```
> cd examples/pytorch/speech-recognition/ASCEND/
> pip install -r requirements.txt
> bash run_train.sh
```
## Model inference
```
python inference.py
```

<!-- ## Demo
```
apt-get install ffmpeg
python demo.py
``` -->
