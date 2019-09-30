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

python PreSumm/src/train.py -task abs -mode train -bert_data_path ~/o3/PreSumm/bert_data/ -dec_dropout 0.2 -model_path ~/o3/PreSumm/models -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 20 -batch_size 8 -train_steps 60 -report_every 10 -accum_count 4 -use_bert_emb true -use_interval true -warmup_steps_bert 10 -warmup_steps_dec 2 -max_pos 512 -visible_gpus -1 -log_file ~/o3/PreSumm/logs/abs_bert_cnndm

 *For colab:
 
!python /content/gdrive/"My Drive"/PreSumm/src/train.py -task abs -mode train -bert_data_path /content/gdrive/"My Drive"/PreSumm/bert_data/ -dec_dropout 0.2 -model_path /content/gdrive/"My Drive"/PreSumm/models -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 20 -batch_size 16 -train_steps 200 -report_every 50 -accum_count 2 -use_bert_emb true -use_interval true -warmup_steps_bert 20 -warmup_steps_dec 10 -max_pos 512 -visible_gpus -1 -log_file /content/gdrive/"My Drive"/PreSumm/logs/abs_bert_cnndm

### Evaluate

python train.py -task abs -mode validate -batch_size 3000 -test_batch_size 500 -bert_data_path BERT_DATA_PATH -log_file ../logs/val_abs_bert_cnndm -model_path MODEL_PATH -sep_optim true -use_interval true -visible_gpus 1 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../logs/abs_bert_cnndm 
 
* `-mode` can be {`validate, test`}, where `validate` will inspect the model directory and evaluate the model for each newly saved checkpoint, `test` need to be used with `-test_from`, indicating the checkpoint you want to use
* `MODEL_PATH` is the directory of saved checkpoints
* use `-mode valiadte` with `-test_all`, the system will load all saved checkpoints and select the top ones to generate summaries (this will take a while)


