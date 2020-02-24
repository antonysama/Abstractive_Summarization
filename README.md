**First things first:**
conda activate p36
pip install --upgrade pip --user

**Outcome:**
Customizes a popular repo by same name. To do abstractive summarization of  one's own text files.

**Methods:**  (Python 3 on Jupyter/ Google Colab) 

Do not 'mount' drive untill you install the dependnecy in step 2 .  ( Under google 'drive' > 'My Drive' is the 'PreSumm' clone. It's the oroginal repo. Under 'bert_data' is pre-preprocessed data. Under 'models' , a pretrained  model. Under 'stanford', the  stanford-core-nlp files. )


Step **1 - Open colab .**  Use a/c & p/w sent separately . 

Training numbers are small for debugging. But, if one needs GPU : Runtime -> Change runtime type -> Hardware Accelerator-> GPU. 

    To cross-check GPU :
    
    #!/usr/bin/env bash
    
    import tensorflow as tf
    
    tf.test.gpu_device_name()   # you should get '/device:GPU:0'


Step **2 - Upload pyrouge .**  It's the ' .ipynb' sent separately. File >  "upload notebook" . Then, Runtime > "run all" . When successfull it says "ran ~11 tests in ~5s."   Mount google drive (use a/c & p/w sent separately) 


Step **3 - Requirements .** In colab use prefix, !

**Set environment to python 36** (e.g., conda activate p36)

    pip install multiprocess --user
    
    pip install tensorboardX --user
    
    pip install pytorch-transformers==1.1.0 --user
        
    pip install torch==1.1.0 --user


Step **4 - stanford-core-nlp .**   Use the bash. Edit the path if needed .

    %%bash
  
    #export CLASSPATH='/content/drive/My Drive/PreSumm/stanford/stanford-corenlp-3.9.2.jar'
  
    export CLASSPATH=~/o3/PreSumm/stanford/stanford-corenlp-3.9.2.jar
    
    echo "Please tokenize this text." | java edu.stanford.nlp.process.PTBTokenizer


Skip preprocessing, we're running on pretrained data.
 
 
Step **5 - Test .**  Foll. under src folder. In colab use  shebang .

    python process_test.py ~/o3/PreSumm/raw_data/cnn/test.txt
    
    #!/bin/python

**note: clean out .pt files under raw data/cnn/bert, logs and.pt files , except model_step_148000.pt**

    python train.py -task abs -mode test -test_from ../models/model_step_148000.pt -batch_size 3000 -test_batch_size 50 -bert_data_path ../bert_data/cnndm -log_file ../logs/val_abs_bert_cnndm -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../logs/abs_test_result

