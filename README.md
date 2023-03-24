# ACAT
Official Pytorch repository for the paper “ACAT: Adversarial Counterfactual Attention for Classification and Detection in Medical Imaging”

Data can be stored in the *data* folder.

Arguments can be modified in the json file: *experiment_config_files/args.json*.

In  order to train the baseline model, you can run:
```
python train.py -filepath_to_arguments_json_config experiment_config_files/args.json -model baseline -experiment_name baseline
```
To generate counterfactual examples and save the saliency maps, you can run:
```
python saliency_maps.py -filepath_to_arguments_json_config experiment_config_files/args.json
```
Saliency maps will be stored in the *saliency_maps* folder.

In order to train ACAT, you can run:
```
python train_two_branches.py -filepath_to_arguments_json_config experiment_config_files/args.json -model ACAT -experiment_name ACAT -resume_from_baseline true -max_epochs 100
```