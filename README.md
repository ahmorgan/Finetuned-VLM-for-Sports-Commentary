# Finetuned-VLM-for-Sports-Commentary

## Tasks

#### Dataset/DataLoader
  (Darbis) DONE Dataset class completed and integrated into training loop.

#### Model training code
  (Andrew) DONE Write basic VLM training script that uses above Dataset class. 

#### Model evaluation code
  (Josh and Andrew) DONE Evaluate model loss/metrics on validation set of unseen points at intervals throughout training. Josh - worked on compute_metrics and evaluation basics, Andrew - added ROUGE eval and inference on random samples every 10 evaluation steps

#### Experiment tracking / results collection
  (Ritvik and Andrew) DONE Andrew - Design experiments, including variations on model type/size and training setup. 
  Ritvik - Run training/eval experiments, including the variations above, and collect results. Also, helping visualize the results.

#### Model inference demo for presentation (Tech Demo)
  (Darbis, Andrew) DONE Inference demo completed.

#### Presentation
  (Tahiyat) DONE Collect and organize visualizations of training (e.g. loss curves).
  (Tahiyat) DONE Write presentation, including experimental setup and results.

# Running inference code

You will need to create a new conda environment using the provided environment.yml file. You will need to also update slurm_script.sh to execute vlm_inference.py (not vlm_training.py). There is a boolean parameter that controls whether or not the checkpoint is used. You may also need to update the checkpoint name in the code to reflect the name of the provided checkpoint folders.

# Running training code

The same instructions as above apply, besides the checkpoint. You can modify the boolean parameters at the top of the file to control which experiment is run. Set neither to true to run the LLM-only attention weights experiment.

All code execution was done via a SLURM computing cluster.

Please contact Andrew (amorga94@charlotte.edu) with questions or difficulties in running the code.
