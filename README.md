# Environment Setup
1. Install anaconda/miniconda
2. Modify conda path in environment.yaml ($CONDA_PATH/envs/FedSSL)
3. ```conda env create --file environment.yaml```
4. ```conda activate FedSSL```

# Run
1. Pass arguments (see options.py)
* ```python main.py --args```

2. Run Tensorboard for monitoring
* ```sh run_tb.sh "Tensorboard log path"```
