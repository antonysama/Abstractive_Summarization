# PreSumm

**This code is for EMNLP 2019 paper [Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345)**

Results on CNN/DailyMail (20/8/2019):


<table class="tg">
  <tr>
    <th class="tg-0pky">Models</th>
    <th class="tg-0pky">ROUGE-1</th>
    <th class="tg-0pky">ROUGE-2</th>
    <th class="tg-0pky">ROUGE-L</th>
  </tr>
  <tr>
    <td class="tg-c3ow" colspan="4">Extractive</td>
  </tr>
  <tr>
    <td class="tg-0pky">TransformerExt</td>
    <td class="tg-0pky">40.90</td>
    <td class="tg-0pky">18.02</td>
    <td class="tg-0pky">37.17</td>
  </tr>
  <tr>
    <td class="tg-0pky">BertSumExt</td>
    <td class="tg-0pky">43.23</td>
    <td class="tg-0pky">20.24</td>
    <td class="tg-0pky">39.63</td>
  </tr>
  <tr>
    <td class="tg-0pky">BertSumExt (large)</td>
    <td class="tg-0pky">43.85</td>
    <td class="tg-0pky">20.34</td>
    <td class="tg-0pky">39.90</td>
  </tr>
  <tr>
    <td class="tg-baqh" colspan="4">Abstractive</td>
  </tr>
  <tr>
    <td class="tg-0lax">TransformerAbs</td>
    <td class="tg-0lax">40.21</td>
    <td class="tg-0lax">17.76</td>
    <td class="tg-0lax">37.09</td>
  </tr>
  <tr>
    <td class="tg-0lax">BertSumAbs</td>
    <td class="tg-0lax">41.72</td>
    <td class="tg-0lax">19.39</td>
    <td class="tg-0lax">38.76</td>
  </tr>
  <tr>
    <td class="tg-0lax">BertSumExtAbs</td>
    <td class="tg-0lax">42.13</td>
    <td class="tg-0lax">19.60</td>
    <td class="tg-0lax">39.18</td>
  </tr>
</table>

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
[Not sure how to do this]

```
### 3.  Json to PyTorch 

```
python PreSumm/src/preprocess.py -mode format_to_bert -raw_path ~/o3/PreSumm/raw_data/. -save_path ~/o3/PreSumm/bert_data  -lower -n_cpus 1 -log_file ~/o3/PreSumm/logs/preprocess.log

```

## Model Training

**First run: For the first time, you should use single-GPU, so the code can download the BERT model. Use ``-visible_gpus -1``, after downloading, you could kill the process and rerun the code with multi-GPUs.**

### Extractive Setting

```
python train.py -task ext -mode train -bert_data_path BERT_DATA_PATH -ext_dropout 0.1 -model_path MODEL_PATH -lr 2e-3 -visible_gpus 0,1,2 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -train_steps 50000 -accum_count 2 -log_file ../logs/ext_bert_cnndm -use_interval true -warmup_steps 10000 -max_pos 512
```

### Abstractive Setting

#### BertAbs
```
python train.py  -task abs -mode train -bert_data_path BERT_DATA_PATH -dec_dropout 0.2  -model_path MODEL_PATH -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0,1,2,3  -log_file ../logs/abs_bert_cnndm
```
#### BertExtAbs
```
python train.py  -task abs -mode train -bert_data_path BERT_DATA_PATH -dec_dropout 0.2  -model_path MODEL_PATH -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0,1,2,3 -log_file ../logs/abs_bert_cnndm  -load_from_extractive EXT_CKPT   
```
* `EXT_CKPT` is the saved `.pt` checkpoint of the extractive model.




## Model Evaluation
```
 python train.py -task abs -mode validate -batch_size 3000 -test_batch_size 500 -bert_data_path BERT_DATA_PATH -log_file ../logs/val_abs_bert_cnndm -model_path MODEL_PATH -sep_optim true -use_interval true -visible_gpus 1 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../logs/abs_bert_cnndm 
```
* `-mode` can be {`validate, test`}, where `validate` will inspect the model directory and evaluate the model for each newly saved checkpoint, `test` need to be used with `-test_from`, indicating the checkpoint you want to use
* `MODEL_PATH` is the directory of saved checkpoints
* use `-mode valiadte` with `-test_all`, the system will load all saved checkpoints and select the top ones to generate summaries (this will take a while)


