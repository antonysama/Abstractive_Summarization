**Steps to run PreSumm on Colab & Drive**   (python 3) 

Do not 'mount' google drive untill you install the dependnecy inn STEP 2.  ( Under google 'drive' > 'My Drive' is the 'PreSumm' cole. It's the oroginal repo. Under 'bert_data' is pre-preprocessed data. Under 'models' , a pretrained  model. Under 'stanford', the  stanford-core-nlp files. )


**STEP 1 - Open colab** using the account id & p/w sent by messenger. 

Training numbers are small for debugging. Butm if colab GPU is needed : Runtime -> Change runtime type -> Hardware Accelerator-> GPU. 

    To cross-check that colab GPU is running:
    
    #!/usr/bin/env bash
    
    import tensorflow as tf
    
    tf.test.gpu_device_name()   # you should get '/device:GPU:0'


**The Set Up :** (STEPS 2-4)


**STEP 2 - Upload pyrouge**  It's the ' .ipynb' sent separately. File >  "upload notebook" . Then, Runtime > "run all" . When successfull it says "ran ~11 tests in ~5s."   After that, mount google drive (useing user & pw sent separately) 


**STEP 3 - Requirements:** 

    !pip install multiprocess
    
    !pip install tensorboardX
    
    !pip install pytorch-transformers==1.1.0
        
    !pip install torch==1.1.0


**STEP 4 - stanford-core-nlp:** - Use the bash. Edit the path if needed 

    %%bash
  
    export CLASSPATH='/content/drive/My Drive/PreSumm/stanford/stanford-corenlp-3.9.2.jar'
  
    echo "Please tokenize this text." | java edu.stanford.nlp.process.PTBTokenizer


Skip the preprocessing, as we're running on pretrained .
 
 
**STEP 5 - Train**  Use  the below shebang (#!...) 

EXTractive model

#!/bin/python

!python '/content/drive/My Drive/PreSumm/src/train.py' -task ext -mode train -bert_data_path 'content/drive/My Drive/PreSumm/bert_data/cnndm' -ext_dropout 0.1 -model_path 'content/drive/My Drive/PreSumm/models' -lr 2e-3 -visible_gpus -1 -report_every 10 -save_checkpoint_steps 10 -batch_size 8 -train_steps 30 -accum_count 2 -log_file 'drive/My Drive/PreSumm/logs/ext_bert_cnndm' -use_interval true -warmup_steps 10 -max_pos 512




