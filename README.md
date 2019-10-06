**Python 3.6**: 

(When running on stand-alone computer, I run 'conda activate p36')

**1-Set Up**

- I have cloned 'nlpyang/PreSumm' on colab (under google). And, I set up pyrouge and other requirements. I downloaded and unzipped both the both the stanford cuore nlp files into the'stanford' folder. Finally, I downloaded the preprocessed  (.pt) files onto  'bert_data.' and the pretrained model under thr 'modeld' folder. 

- At this point, get Stanford CoreNLP started and skip to stage 3.

  %%bash
  export CLASSPATH='/content/content/My Drive/PreSumm/stanford/stanford-corenlp-3.9.2.jar'
  echo "Please tokenize this text." | java edu.stanford.nlp.process.PTBTokenizer

**2-Pre-process**

- At this point, I skip preprocessing. 
 
**3-Train, Val, and Test** 
**For the first run use debugging numbers. Thereafter, check the original repo for larger numbers**

Download and unzip pretrained model into PreSumm/models https://drive.google.com/uc?id=1-IKVCtc4Q-BdZpjXc4s70_fRsWnjtYLr&export=download

EXtractive model

python PreSumm/src/train.py -task ext -mode train -bert_data_path ~/o3/PreSumm/bert_data/cnndm -ext_dropout 0.1 -model_path ~/o3/PreSumm/models -lr 2e-3 -visible_gpus -1 -report_every 10 -save_checkpoint_steps 10 -batch_size 8 -train_steps 30 -accum_count 2 -log_file ~/o3/PreSumm/logs/ext_bert_cnndm -use_interval true -warmup_steps 10 -max_pos 512

ABSstractive model

python PreSumm/src/train.py -task abs -mode train -bert_data_path ~/o3/PreSumm/bert_data/cnndm -dec_dropout 0.2 -model_path ~/o3/PreSumm/models -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 10 -batch_size 8 -train_steps 30 -report_every 10 -accum_count 2 -use_bert_emb true -use_interval true -warmup_steps_bert 5 -warmup_steps_dec 2 -max_pos 512 -visible_gpus -1 -log_file ~/o3/PreSumm/logs/abs_bert_cnndm

EXABS model

python PreSumm/src/train.py  -task abs -mode train -bert_data_path ~/o3/PreSumm/bert_data/cnndm -dec_dropout 0.2  -model_path ~/o3/PreSumm/models -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 10 -batch_size 8 -train_steps 30 -report_every 10 -accum_count 2 -use_bert_emb true -use_interval true -warmup_steps_bert 5 -warmup_steps_dec 2 -max_pos 512 -visible_gpus -1 -log_file ~/o3/PreSumm/logs/abs_bert_cnndm  -load_from_extractive ~/o3/PreSumm/models/model_step_30.pt **

**load the saved '.pt' checkpoint of the extractive model.

### Evaluate
python PreSumm/src/train.py -task abs -mode validate -test_all -batch_size 8 -test_batch_size 2 -bert_data_path ~/o3/PreSumm/bert_data/cnndm -log_file ~/o3/PreSumm/logs/val_abs_bert_cnndm -model_path ~/o3/PreSumm/models -sep_optim true -use_interval true -visible_gpus -1 -max_pos 512 -max_length 10 -alpha 0.95 -min_length 5 -result_path ~/o3/PreSumm/logs/abs_bert_cnndm

##Gives folowing error:
##RuntimeError: Error(s) in loading state_dict for AbsSummarizer:
##solns: https://github.com/amdegroot/ssd.pytorch/issues/342 & https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/8

### Test
python PreSumm/src/train.py -task abs -mode test -test_from ~/o3/PreSumm/models/model_step_148000.pt -batch_size 16 -test_batch_size 2 -bert_data_path ~/o3/PreSumm/bert_data/cnndm -log_file ~/o3/PreSumm/logs/val_abs_bert_cnndm -sep_optim true -use_interval true -visible_gpus -1 -max_pos 512 -max_length 10 -alpha 0.95 -min_length 5 -result_path ~/o3/PreSumm/logs/abs_bert_cnndm 

* To enable GPU backend for your notebook. Runtime->Change runtime type->Hardware Accelerator->GPU. To cross-check whether the GPU is enabled you can run 'import tensorflow as tf' . Then, run 'tf.test.gpu_device_name()' .
* `-mode` can be {`validate, test`}, where `validate` will inspect the model directory and evaluate the model for each newly saved checkpoint, `test` need to be used with `-test_from`, indicating the checkpoint you want to use
* `MODEL_PATH` is the directory of saved checkpoints
* use `-mode valiadte` with `-test_all`, the system will load all saved checkpoints and select the top ones to generate summaries (this will take a while)


