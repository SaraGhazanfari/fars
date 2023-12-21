# FaRS: Fast Randomized Smoothing

<p align="justify">  In this project, we aim to propose a Fast version of Randomized Smoothing, FaRS, by decreasing the time of the Monte Carlo sampling performed to estimate the probabilities and calculate the certified bound. The number of samplings cannot be reduced due to the confidence level required for the estimations. However, by posing the Lipschitz assumption on the model, we could limit the Monte Carlo sampling to the last linear layers performed on the features (in the feature space) to predict classes. More precisely FaRS is composed of a 1-Lipschitz backbone and performs Monte Carlo sampling only on the feature space. In the current project, we focused on a model with a one-layer linear module, which cancels the need for the Monte Carlo sampling. However, in general for the Linear Module to have more expressivity we need more than one linear layer, and the Monte Carlo sampling is needed to generate the certified radius. In order to have bigger margins for each data point we used two techniques that showed its effectiveness in the experiments. </p>

The following command can be used to train the classifier:
```
python3 -m fars.main --dataset imagenette --model-name small --train_dir path/to/backbone/checkpoint --data_dir path/to/data --batch_size 32 --epochs 100 --save_checkpoint_epochs 10 --num_linear  1"
```
