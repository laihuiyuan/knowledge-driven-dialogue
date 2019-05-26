Knowledge-driven Dialogue
=============================

This is a pytorch implementation of generative-based model for knowledge-driven dialogue. The original version comes from Baidu.com, https://github.com/baidu/knowledge-driven-dialogue

## Final Score
### 35th, F1=40.62,BLEU1/BLEU2=0.370/0.235, DISTINCT1/DISTINCT2=0.094/0.240.

## Requirements

* cuda=9.0
* cudnn=7.0
* python>=3.6
* pytorch>=1.0
* nltk
* tqdm
* numpy
* scikit-learn

## Quickstart

### Step 1: Preprocess the data

Put the data provided by the organizer under the data folder and rename them  train/dev/test.txt: 

```
./data/resource/train.txt
./data/resource/dev.txt
./data/resource/test.txt
```

### Step 2: Train the model

Train model with the following commands.

```bash
sh run_train.sh
```

### Step 3: Test the Model

Test model with the following commands.

```bash
sh run_test.sh
```

