# Abstractive Summary

**Python version**: This code is in Python3.6

cd o3

conda activate p36

**Requirements**: 

#torch==1.1.0, pytorch_transformers tensorboardX multiprocess pyrouge

* For colab modify to !pip

pip install multiprocess

#pip install tensorboard==1.0.0a6

pip install tensorboardX

pip install pytorch-transformers==1.1.0

pip install torch==1.1.0 --user

**Updates**: For encoding a text longer than 512 tokens, for example 800. Set max_pos to 800 during both preprocessing and training.
##  Preporcess
### 1.  Connecte to the pre-downloaded Stanford CoreNLP & test it

export CLASSPATH=~/o3/stanford/stanford-corenlp-3.9.2.jar

echo "Please tokenize this text." | java edu.stanford.nlp.process.PTBTokenizer

### 2.   Split sentences and tokenize
python PreSumm/src/preprocess.py  -mode tokenize  -raw_path ~/o3/PreSumm/raw_data/ -save_path ~/o3/PreSumm/json_data/  -log_file ~/o3/PreSumm/logs/cnndm.log
### 3.  To Simple Json 
python PreSumm/src/preprocess.py  -mode format_to_lines  -raw_path ~/o3/PreSumm/json_data/  -save_path ~/o3/PreSumm/json_data2/  -n_cpus 1  -use_bert_basic_tokenizer false  -map_path ~/o3/PreSumm/urls/  -log_file ~/o3/PreSumm/logs/cnndm.log
### 4.  To PyTorch (& rename as 'train.pt')
python PreSumm/src/preprocess.py -mode format_to_bert -raw_path ~/o3/PreSumm/json_data/ -save_path ~/o3/PreSumm/bert_data/  -lower -n_cpus 1 -log_file ~/o3/PreSumm/logs/preprocess.log 

mv ~/o3/PreSumm/bert_data/*.pt ~/o3/PreSumm/bert_data/train.pt # rename file to train.pt

###  Train 
**For the first run use debugging numbers. Thereafter, check the original repo for larger numbers**

EX

python PreSumm/src/train.py -task ext -mode train -bert_data_path ~/o3/PreSumm/bert_data/ -ext_dropout 0.1 -model_path ~/o3/PreSumm/models -lr 2e-3 -visible_gpus -1 -report_every 10 -save_checkpoint_steps 10 -batch_size 8 -train_steps 30 -accum_count 2 -log_file ~/o3/PreSumm/logs/ext_bert_cnndm -use_interval true -warmup_steps 10 -max_pos 512

ABS

python PreSumm/src/train.py -task abs -mode train -bert_data_path ~/o3/PreSumm/bert_data/ -dec_dropout 0.2 -model_path ~/o3/PreSumm/models -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 10 -batch_size 8 -train_steps 30 -report_every 10 -accum_count 2 -use_bert_emb true -use_interval true -warmup_steps_bert 5 -warmup_steps_dec 2 -max_pos 512 -visible_gpus -1 -log_file ~/o3/PreSumm/logs/abs_bert_cnndm

EXABS

python PreSumm/src/train.py  -task abs -mode train -bert_data_path ~/o3/PreSumm/bert_data/ -dec_dropout 0.2  -model_path ~/o3/PreSumm/models -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 20 -batch_size 8 -train_steps 60 -report_every 20 -accum_count 4 -use_bert_emb true -use_interval true -warmup_steps_bert 5 -warmup_steps_dec 2 -max_pos 512 -visible_gpus -1 -log_file ~/o3/PreSumm/logs/abs_bert_cnndm  -load_from_extractive EXT_CKPT   


 *For colab: see repo colab

### Evaluate
python PreSumm/src/train.py -task abs -mode validate -batch_size 16 -test_batch_size 2 -bert_data_path ~/o3/PreSumm/bert_data/ -log_file ~/o3/PreSumm/logs/val_abs_bert_cnndm -model_path ~/o3/PreSumm/models -sep_optim true -use_interval true -visible_gpus -1 -max_pos 512 -max_length 10 -alpha 0.95 -min_length 5 -result_path ~/o3/PreSumm/logs/abs_bert_cnndm 

### Test
python PreSumm/src/train.py -task abs -mode test -test_from ~/o3/PreSumm/bert_data/model_step_148000.pt -batch_size 16 -test_batch_size 2 -bert_data_path ~/o3/PreSumm/bert_data/ -log_file ~/o3/PreSumm/logs/val_abs_bert_cnndm -sep_optim true -use_interval true -visible_gpus -1 -max_pos 512 -max_length 10 -alpha 0.95 -min_length 5 -result_path ~/o3/PreSumm/logs/abs_bert_cnndm 
 
* `-mode` can be {`validate, test`}, where `validate` will inspect the model directory and evaluate the model for each newly saved checkpoint, `test` need to be used with `-test_from`, indicating the checkpoint you want to use
* `MODEL_PATH` is the directory of saved checkpoints
* use `-mode valiadte` with `-test_all`, the system will load all saved checkpoints and select the top ones to generate summaries (this will take a while)


