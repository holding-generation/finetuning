# Judicial Holding Extraction and Evaluation from Longform Legal Documents

This repository contains python code for training and inference on Long T5 and Longformer, which serve as some of our principle experiments on this project.
It also contains code calculating evaluation metrics ROUGE, SummaC, and Bleurt. It's principle author is Jesse Woo (jw4202). 

For code on data engineering (downloading, preprocessing), and inference on baselines, see the Local-LLM-Code repository that sits under the same holding-generation github 
organization as this repo. The principle author of the code in that repository is Lawrence Leung (lsl2162)

## Execution Paths for Notebooks:
* All notebooks can be run in google Colab.
* Dependencies are pip installed directly in each notebook.
* A shared google drive is mounted to the notebook and filepaths are given directly to that drive.
* Input data for the models should have the shape of a csv file with 'input' and 'output' columns.
* Inputs for SummaC should be a csv with 'Input', 'Prediction', and 'Reference' columns.

## Execution Paths for Finetuning and BLEURT
* Finetuning scripts should run in a GCP VM with the standard deep learning disk image and an L4 GPU.
* Input data for training the models should have the shape of a csv file with 'input' and 'output' columns.
* pip install -r requirements.txt for dependencies
* to run training scripts:
    - python finetune_<model_name>.py cleaned_train_dataset.csv cleaned_val_dataset.csv cleaned_test_dataset.csv
    - I also used the nohup command to run training in the background.
* to run prediction script:
    - python predict.py cleaned_test_dataset.csv <path_to_model>
    I also used the nohup command to run inference in the background.
* to run data formatting script for BLEURT:
    - python convert_to_json_bleurt.py <model_name>_predictions.csv <desired_file_name>.jsonl
* to calculate BLEURT Scores, use the guide found in the official repo https://github.com/google-research/bleurt
    - pip install --upgrade pip  # ensures that pip is current
    - git clone https://github.com/google-research/bleurt.git
    - cd bleurt
    - pip install .
    - wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
    - unzip BLEURT-20.zip
    - python -m bleurt.score_files -sentence_pairs_file=<desired_file_name>.jsonl -bleurt_checkpoint=BLEURT-20 -score_files=<model_name>_bleurt_scores.txt