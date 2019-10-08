**Steps to run PreSumm** on Colab & Drive  (python 3) 

Do not 'mount' drive untill you install the dependnecy in step 2 .  ( Under google 'drive' > 'My Drive' is the 'PreSumm' clone. It's the oroginal repo. Under 'bert_data' is pre-preprocessed data. Under 'models' , a pretrained  model. Under 'stanford', the  stanford-core-nlp files. )


Step **1 - Open colab .**  Use a/c & p/w sent separately . 

Training numbers are small for debugging. But, if one needs GPU : Runtime -> Change runtime type -> Hardware Accelerator-> GPU. 

    To cross-check GPU :
    
    #!/usr/bin/env bash
    
    import tensorflow as tf
    
    tf.test.gpu_device_name()   # you should get '/device:GPU:0'


Step **2 - Upload pyrouge .**  It's the ' .ipynb' sent separately. File >  "upload notebook" . Then, Runtime > "run all" . When successfull it says "ran ~11 tests in ~5s."   Mount google drive (use a/c & p/w sent separately) 


Step **3 - Requirements .** 

    !pip install multiprocess
    
    !pip install tensorboardX
    
    !pip install pytorch-transformers==1.1.0
        
    !pip install torch==1.1.0


Step **4 - stanford-core-nlp .**   Use the bash. Edit the path if needed .

    %%bash
  
    export CLASSPATH='/content/drive/My Drive/PreSumm/stanford/stanford-corenlp-3.9.2.jar'
  
    echo "Please tokenize this text." | java edu.stanford.nlp.process.PTBTokenizer


Skip preprocessing, we're running on pretrained data.
 
 
Step **5 - Train .**  Use  the  shebang .


#!/bin/python

!python '/content/drive/My Drive/PreSumm/src/train.py' -task ext -mode train -bert_data_path 'content/drive/My Drive/PreSumm/bert_data/cnndm' -ext_dropout 0.1 -model_path 'content/drive/My Drive/PreSumm/models' -lr 2e-3 -visible_gpus -1 -report_every 10 -save_checkpoint_steps 10 -batch_size 8 -train_steps 30 -accum_count 2 -log_file 'drive/My Drive/PreSumm/logs/ext_bert_cnndm' -use_interval true -warmup_steps 10 -max_pos 512


**Error** screenshots are in a drive folder.

