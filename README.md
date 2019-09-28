# PreSumm

**This code is for EMNLP 2019 paper [Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345)**

**Python version**: This code is in Python3.6

cd o3

conda activate p36

**Requirements**: 

#torch==1.1.0, pytorch_transformers tensorboardX multiprocess pyrouge

pip install multiprocess

pip install tensorboard

pip install pytorch-transformers==1.1.0

pip install torch-0.1.10.post1-cp36-cp36m-linux_x86_64.whl

python PreSumm/src/preprocess.py -mode format_to_bert -raw_path ~/o3/PreSumm/raw_data/. -save_path ~/o3/PreSumm/bert_data  -lower -n_cpus 1 -log_file ~/o3/PreSumm/logs/preprocess.log

**Updates**: For encoding a text longer than 512 tokens, for example 800. Set max_pos to 800 during both preprocessing and training.

### 1.  Connecte to the pre-downloaded Stanford CoreNLP & test it

export CLASSPATH=~/o3/stanford/stanford-corenlp-3.9.2.jar

echo "Please tokenize this text." | java edu.stanford.nlp.process.PTBTokenizer

### 2.   Sentence Splitting and Tokenization

```
python PreSumm/src/preprocess.py -mode tokenize -raw_path ~/o3/PreSumm/raw_data/ -save_path ~/o3/PreSumm/json_data/ -log_file ~/o3/PreSumm/logs/cnndm.log

```
### 3.  Json to PyTorch 

```
python PreSumm/src/preprocess.py -mode format_to_bert -raw_path ~/o3/PreSumm/json_data/. -save_path ~/o3/PreSumm/bert_data  -lower -n_cpus 1 -log_file ~/o3/PreSumm/logs/preprocess.log

```

## Model Training

**First run: For the first time, you should use single-GPU, so the code can download the BERT model. Use ``-visible_gpus -1``, after downloading, you could kill the process and rerun the code with multi-GPUs.**

### Abstractive Setting

#### BertAbs
```
python PreSumm/src/train.py  -task abs -mode train -bert_data_path ~/o3/PreSumm/bert_data -dec_dropout 0.2  -model_path ~/o3/PreSumm/models -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 10 -train_steps 4 -report_every 2 -accum_count 1 -use_bert_emb true -use_interval true -warmup_steps_bert 4 -warmup_steps_dec 2 -max_pos 800 -visible_gpus 1  -log_file ~/o3/PreSumm/logs/abs_bert_cnndm
```

## Model Evaluation
```
 python train.py -task abs -mode validate -batch_size 3000 -test_batch_size 500 -bert_data_path BERT_DATA_PATH -log_file ../logs/val_abs_bert_cnndm -model_path MODEL_PATH -sep_optim true -use_interval true -visible_gpus 1 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../logs/abs_bert_cnndm 
```
* `-mode` can be {`validate, test`}, where `validate` will inspect the model directory and evaluate the model for each newly saved checkpoint, `test` need to be used with `-test_from`, indicating the checkpoint you want to use
* `MODEL_PATH` is the directory of saved checkpoints
* use `-mode valiadte` with `-test_all`, the system will load all saved checkpoints and select the top ones to generate summaries (this will take a while)


