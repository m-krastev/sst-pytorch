# Practical 2 - Sentiment Analysis
The goal of this practical is to implement a sentiment analysis model using PyTorch. The model will be trained on the Stanford Sentiment Treebank (SST) dataset.

## Code Structure

- `bow|cbow|deepcbow|lstm|treelstm.py`: Main script to run the model. Contains the model definition, training and evaluation code. You can run the script and look at the command line arguments to see how to run the model.
- `utils.py` : Contains utility functions for loading the data, plotting, etc. Contains the code for the `TreeDataset` class which is used to load the data.
- `trainingutils.py` : Contains utility functions for training and evaluation, deprecated.
- `run_tree.job`: Slurm job file to run the model on the cluster (with 3 different seeds for reproducibility).


## Command Line Arguments

- `--device`: Specifies the device to use for computation. Default is "cuda" if CUDA is available, otherwise "cpu".
- `--epochs`: Number of epochs for training. Default is 20.
- `--batch_size`: Batch size for training. Default is 32.
- `--lr`: Learning rate for the optimizer. Default is 5e-4.
- `--seed`: Seed for random number generation. Default is 42.
- `--model_dir`: Directory to save the model file. Default is "save/".
- `--num_workers`: Number of workers for the dataloader. Default is the number of CPUs.
- `--plot_dir`: Directory to save the plots. Default is "plots/".
- `--data_dir`: Directory for the data. Default is "data/".
- `--results_dir`: Directory to save the results. Default is "results/".
- `--evaluate`: Flag to specify whether to only evaluate the model. Default is False.
- `--load_model` (deprecated): Name of the model file to load. Default is None.
- `--hidden_dim`: Dimension of the hidden layer. Applies for LSTM. Default is 150.
- `--hidden_dims`: Dimensions of the hidden layers.Applies for DeepCBOW. Default is [100, 100].
- `--embeddings_type`: Type of embeddings to use. Choices are "word2vec" and "glove". Default is "word2vec".
- `--num_iterations` (deprecated, use epochs): Number of iterations for training. Default is 10000.
- `--embedding_dim`: Dimension of the embeddings. Applies for CBOW and DeepCBOW. Default is 300.
- `--print_every` (deprecated, unused): Frequency of printing training status. Default is 1000.
- `--eval_every` (deprecated, unused): Frequency of evaluating the model. Default is 1000.
- `--classes`: Classes for classification. Default is ["very negative", "negative", "neutral", "positive", "very positive"].
- `--debug`: Flag to specify whether to run in debug mode. Debug mode turns off progress bars and some printing utilities. Default is False.
- `--lower`: Flag to specify whether to lowercase the data. Default is True.
- `--train_embeddings`: Flag to specify whether to train embeddings along with the model. Default is False.
- `--checkpoint`: Path of the model checkpoint file to load. Default is None.
