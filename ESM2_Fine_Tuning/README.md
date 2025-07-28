# Fine Tuning Protein Language Model ESM-2 with Torch

This contains a couple of scripts which prepare a single protein sequence for training of masked sequence prediction for the amyloid beta protein sequence. ```esm_finetune.py``` modifies the weights of the base ESM2 model while ```esm_lora_finetune.py``` uses parameter efficient fine tuning (PEFT) with the Low Rank Adaptation (LoRA) method.

The training scripts only train for a dataset of one masked sequence but can be easily extended to larger datasets. These small examples are just for me to explore the workflows of working with these language models.
