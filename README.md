* Install conda
	
* Setup conda environment with environment.yaml
	* Need to set "prefix" to your conda envs directory before creating the conda environment
	* `conda env create -f environment.yaml`

* After creating the environment, activate it
	* `conda activate FedSSL`

* wandb setup 
	* `wandb login` after signing up at the site https://wandb.ai/site
	* `wandb init` then enter the API Key

* 'scripts' folder contains sample scripts for iid and non-iid FL training
	* `sbatch FL_iid.sh`




