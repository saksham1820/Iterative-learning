## Iterative Segmentation

This is the code used to perform the experiment of iterative learning .

The code uses the well-preprocessed `ACDC` dataset for reviewers. The dataset should be keep private based on the dataset agreement.

Our code is based on `deepclustering2` package, which is a personal research framework. It will automatically install all dependency on a conda virtual environment and without resorting to `requirement.txt`.


-----------------
##### Basic script for setting a conda-based virtual environment.
```bash
conda create -p ./venv python=3.7

conda activate ./venv

conda install pytorch ==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch  # install pytorch 1.6.0
pip install deepclustering2-2.0.0-py3-none-any.whl
python setup.py install  
# all packages should be set properly automatically.
```
In case of failure of running the experiments, please refer to `requirement.txt` to see the packages

----------------
##### Basic script to start training 
```bash
# for num_iter = 1
python main.py Arch.input_dim=5 Iterations.num_iter=1 Aggregator.alpha=0.9
# for num_iter = 2
python main.py Arch.input_dim=5 Iterations.num_iter=2 Aggregator.alpha=0.9
# for num_iter = 3
python main.py Arch.input_dim=5 Iterations.num_iter=3 Aggregator.alpha=0.9
# for num_iter = 4
python main.py Arch.input_dim=5 Iterations.num_iter=4 Aggregator.alpha=0.9
# for num_iter = 5
python main.py Arch.input_dim=5 Iterations.num_iter=5 Aggregator.alpha=0.9  

```
One can change the parameters on the cmd if needed.
Please refer to the default configuration in `config/semi.yaml` all set of controllable hyperparameters. All of them can be changed using cmd as above.


---------------------
##### Performance
Yet to be confirmed.



