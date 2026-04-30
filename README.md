# Finetuned-VLM-for-Sports-Commentary

## Tasks

#### Dataset/DataLoader
  (Darbis) DONE Dataset class completed and integrated into training loop.

#### Model training code
  (Andrew) DONE Write basic VLM training script that uses above Dataset class. 

#### Model evaluation code
  (Josh and Andrew) DONE Evaluate model loss/perplexity and on validation set of unseen points at intervals throughout training. Josh - worked on compute_metrics and evaluation basics, Andrew - added ROUGE eval and inference on random samples every evaluation steps

#### Experiment tracking / results collection
  Design experiments, including variations on model type/size and evaluation scenarios (e.g. try different types of tennis points).
  Run training/eval experiments, including the variations above, and collect results.

#### Model inference demo for presentation (Tech Demo)
  (Andrew, Darbis) DONE Inference demo completed. Can easily adapt this for tennis point inference.

#### Presentation
  (Tahiyat) Collect and organize visualizations of training (e.g. loss curves).
  (Tahiyat) Write presentation, including experimental setup and results.

# Running inference code

You will need to create a new conda environment using the provided environment.yml file. You will need to also update slurm_script.sh to execute vlm_inference.py (not vlm_training.py). There is a boolean parameter that controls whether or not the checkpoint is used. You may also need to update the checkpoint name in the code.

# Running training code

The same instructions as above apply, besides the checkpoint. You can modify the boolean parameters at the top of the file to control which experiment is run. Set neither to true to run the LLM-only attention weights experiment.
