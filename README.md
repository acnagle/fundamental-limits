# Create the conda environment

``` bash
conda env create -f environment.yml
```
# Reproduce results for the prompt compressors

The experiments in this step must be run before running the code for the optimal R-D curves.

## Synthetic dataset
Navigate to the `toy_dataset` folder.

Launch the training runs to fine-tune the models with 

``` bash
bash finetune.sh
```

Now gather the prompt compression results by issuing

``` bash
bash inference.sh
```

## NLP datasets

### Training
Navigate to the `nlp_dataset/LLMLingua/experiments/llmlingua2/data_collection` folder and run

``` bash
bash collect_data.sh
```

to form the dataset for training the QuerySelect and Adaptive QuerySelect encoder model. Then, launch the training script in `nlp_dataset/LLMLingua/experiments/llmlingua2/model_training` with

``` bash
bash train.sh
```

Alternatively, the model weights are available on [Hugging Face](https://huggingface.co/acnagle/QuerySelect)

### Inference
Navigate to the `nlp_dataset` folder.

Run the experiments for the prompt compression methods by running

``` bash
bash inference.sh
```

Experiments for NarrativeQA can be run with

``` bash
bash narrativeqa.sh
```

However, running `narrativeqa.sh` will take an especially long time. It is recommended to scatter the jobs across multiple GPUs. The beam search job alone takes approximately 1200 A100 GPU hours.

# Reproduce the optimal R-D curves
Navigate to the optimal_rd folder.

The file RD_dual_LP_solver.py can be called with the following arguments:

python RD_dual_LP_solver.py [--data_path DATA_PATH] [--condition CONDITION] [--rate_vals RATE_VALS] [--wandb_project WANDB_PROJECT]

- DATA_PATH is the name of the .jsonl file from the 'datasets' folder with the data needed to compute the distortion-rate function (default is 'Mistral-7B-Instruct-v0.2_optimal.jsonl')
- CONDITION can take 4 values:
	-- 0: computes the query-agnostic curve (default)
	-- 1: computes the average query-aware curve
	-- 2: computes the conditional query-aware curves (one for each query)
	-- 3: does all of the above and stores it in a file 'output_RD/optimal_RD_<DATA_PATH>.json'
- RATE_VALS is the list of rate values at which the distortion-rate curve is to be plotted 
- WANDB_PROJECT is the name of the project if using wandb (default is 'rd-lp-dual-eval')

Examples:

1. To compute the query-agnostic optimal curve for the file 'Mistral-7B-Instruct-v0.2_optimal.jsonl' at certain rate values, run 
python RD_dual_LP_solver.py --data_path 'Mistral-7B-Instruct-v0.2_optimal.jsonl' --condition 0 --rate_vals 0.12 0.13 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95
2. To compute the conditional query-aware optimal curve for the file 'Mistral-7B-Instruct-v0.2_optimal_forced_ft.jsonl' at certain rate values, run 
python RD_dual_LP_solver.py --data_path 'Mistral-7B-Instruct-v0.2_optimal_forced_ft.jsonl' --condition 2 --rate_vals 0.12 0.13 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95
2. To obtain all optimal curves for the file 'Mistral-7B-Instruct-v0.2_accuracy_optimal_ft.jsonl' at certain rate values, run 
python RD_dual_LP_solver.py --data_path 'Mistral-7B-Instruct-v0.2_accuracy_optimal_ft.jsonl' --condition 3 --rate_vals 0.12 0.13 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95
