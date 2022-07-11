## Towards Computationally Feasible Deep Active Learning
A repository to reproduce the experiments from the paper "Towards Computationally Feasible Deep Active Learning".

## Installation
Install the library:
```
pip install -e .
```

## Usage
The `configs` folder contains config files with general settings. The `experiments` folder contains config files with experimental design. To run an experiment with a chosen configuration, specify config file name in `HYDRA_CONFIG_NAME` variable and run `train.sh` script (see `./examples/al` for details). 

For example to launch PLASM on AG-News with ELECTRA as a successor model:
```
cd PATH_TO_THIS_REPO
HYDRA_CONFIG_PATH=../experiments/ag_news HYDRA_EXP_CONFIG_NAME=ag_plasm python active_learning/run_tasks_on_multiple_gpus.py
```

### Config structure explanation
- `cuda_devices`: list of CUDA devices to use: one experiment on one CUDA device. `cuda_devices=[0,1]` means using zero-th and first devices.
- `config_name`: name of config from **configs** folder with general settings: dataset, experiment setting (e.g. LC/ASM/PLASM), model checkpoints, hyperparameters etc.
- `config_path`: path to config with general settings.
- `command`: **.py** file to run. For AL experiments, use **run_active_learning.py**.
- `args`: arguments to modify from a general config in the current experiment. `acquisition_model.name=xlnet-base-cased` means that _xlnet-base-cased_ will be used as an acquisition model.
- `seeds`: random seeds to use. `seeds=[4837, 23419]` means that two separate experiments with the same settings (except for **seed**) will be run: one with **seed == 4837**, one with **seed == 23419**.

### Output explanation
By default, the results will be present in the folder `RUN_DIRECTORY/workdir_run_active_learning/DATE_OF_RUN/${TIME_OF_RUN}_${SEED}_${MODEL_CHECKPOINT}`. For instance, when launching from the repository folder: `al_nlp_feasible/workdir/run_active_learning/2022-06-11/15-59-31_23419_distilbert_base_uncased_bert_base_uncased`.

- When running a classic AL experiment (acquisition and successor models coincide, regardless of using UPS), the file with the model metrics is `acquisition_metrics.json`.
- When running an acquisition-successor mismatch experiment, the file with the model metrics is `successor_metrics.json`.
- When running a PLASM experiment, the file with the model metrics is `target_tracin_quantile_-1.0_metrics.json` (**-1.0** stands for the filtering value, meaning adaptive filtering rate; when using a deterministic filtering rate (e.g. **0.1**), the file will be named `target_tracin_quantile_0.1_metrics.json`). The file with the metrics of the model **without filtering** is `target_metrics.json`.


### Datasets
The research has employed 2 NER datasets (CoNLL-2003, OntoNotes-2012) and 2 Text Classification (CLS) datasets (AG-News, IMDB). If one wants to launch an experiment on a custom dataset, they need to use one of the following ways to add it:

1) Upload to [Hugging Face datasets](https://huggingface.co/datasets) and set: `config.data.path=datasets, config.data.dataset_name=DATASET_NAME, config.data.text_name=COLUMN_WITH_TEXT_OR_TOKENS_NAME, config.data.label_name=COLUMN_WITH_LABELS_OR_NER_TAGS_NAME`
2) Upload to **data/DATASET_NAME** folder, create **train.csv** / **train.json** file with the dataset, and set: `config.data.path=PATH_TO_THIS_REPO/data, config.data.dataset_name=DATASET_NAME, config.data.text_name=COLUMN_WITH_TEXT_OR_TOKENS_NAME, config.data.label_name=COLUMN_WITH_LABELS_OR_NER_TAGS_NAME`
3) \* Upload to **data/DATASET_NAME** **train.txt**, **dev.txt**, and **test.txt** files and set the arguments as in the previous point.
4) \*\* Upload to **data/DATASET_NAME** with each folder for each class, where each file in the folder contains a text with the label of the folder. For details, please see the **bbc_news** dataset in **./data**. The arguments must be set as in the previous two points.

\* - only for NER datasets

\*\* - only for CLS datasets

### Models
The current version of the repository supports all models from [HuggingFace Transformers](https://huggingface.co/models), which can be used with `AutoModelForSequenceClassification` / `AutoModelForTokenClassification` classes (for CLS / NER). For CNN-based / BiLSTM-CRF models, please see the **al_cls_cnn.yaml** / **al_ner_bilstm_crf.yaml** configs from **./configs** folder for details.


## Citation
```
@article{tsvigun-etal-2022-plasm,
  author    = {Akim Tsvigun and
               Artem Shelmanov and
               Gleb Kuzmin and
               Leonid Sanochkin and
               Daniil Larionov and
               Gleb Gusev and
               Manvel Avetisian and
               Leonid Zhukov},
  title     = {Towards Computationally Feasible Deep Active Learning},
  journal   = {CoRR},
  volume    = {abs/2205.03598},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2205.03598},
  doi       = {10.48550/arXiv.2205.03598},
  eprinttype = {arXiv},
  eprint    = {2205.03598},
  timestamp = {Wed, 11 May 2022 17:29:40 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2205-03598.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## License
© 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research Institute" (AIRI). All rights reserved.

Licensed under the GNU GPLv3 License.