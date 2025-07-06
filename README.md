# Deep Imbalanced Regression via Hierarchical Classification Adjustment (HCA)
This repository contains the original pytorch code for our paper "Deep Imbalanced Regression via Hierarchical Classification Adjustment" in CVPR 2024. 


# How to use
- prepare the environment with packages include PyTorch, tensorboard_logger, numpy, pandas, scipy, tqdm, matplotlib, PIL, wget;
- download the imdb-wiki dataset by running the code "python imdbwiki_data/download_imdb_wiki.py";
- download the model in this [link](https://drive.google.com/file/d/1aihIZmxb_psUDPOu79QNt-0IOfP44eJK/view?usp=sharing) and unzip it in the folder;
- run the code with "python main.py". You can try other models by adding the "--resume" config to the code as "python main.py --resume <path_of_model_checkpoint>". 

