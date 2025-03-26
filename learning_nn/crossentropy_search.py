if __name__ == "__main__":
    from nn_layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from nn_losses import CrossEntropyLossLayer
    from nn_optimizer import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from .optimizers import SGDOptimizer
    from .losses import CrossEntropyLossLayer
    from ..train import plot_model_guesses, train

from typing import Any, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

RNG = torch.Generator()
RNG.manual_seed(446)


@problem.tag("hw3-A")
def crossentropy_parameter_search(
    dataset_train: TensorDataset, dataset_val: TensorDataset
) -> Dict[str, Any]:
    """
    Main subroutine of the CrossEntropy problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers
    NOTE: Each model should end with a Softmax layer due to CrossEntropyLossLayer requirement.

    Notes:
        - When choosing the number of epochs, consider effect of other hyperparameters on it.
            For example as learning rate gets smaller you will need more epochs to converge.

    Args:
        dataset_train (TensorDataset): Dataset for training.
        dataset_val (TensorDataset): Dataset for validation.

    Returns:
        Dict[str, Any]: Dictionary/Map containing history of training of all models.
            You are free to employ any structure of this dictionary, but we suggest the following:
            {
                name_of_model: {
                    "train": Per epoch losses of model on train set,
                    "val": Per epoch losses of model on validation set,
                    "model": Actual PyTorch model (type: nn.Module),
                }
            }
    """
    #     - Linear Regression Model
    #     - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
    #     - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
    #     - Network with two hidden layers (each with size 2)
    #         and Sigmoid, ReLU activation function after corresponding hidden layers
    #     - Network with two hidden layers (each with size 2)
    #         and ReLU, Sigmoid activation function after corresponding hidden layers
    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    batch_sizes = [32, 64, 128, 256]

    models = {}
    
    # helper function to record best models
    def record_best_model(model_name, model, train_loss, val_loss, lr, bs, epochs):
        if model_name in models:
  
            current_min_val_loss = min(val_loss)
            previous_min_val_loss = min(models[model_name]["val"])
            
            if current_min_val_loss < previous_min_val_loss:
                models[model_name] = {
                    "train": train_loss,
                    "val": val_loss,
                    "model": model,
                    "lr": lr,
                    "batch_size": bs,
                    "epochs": epochs
                }
        else:
            models[model_name] = {
                "train": train_loss,
                "val": val_loss,
                "model": model,
                "lr": lr,
                "batch_size": bs,
                "epochs": epochs
            }
    
    for lr in learning_rates:
        epochs = int(60 * (0.1 / lr)) 
        epochs = max(60, min(epochs, 200))
        for bs in batch_sizes:
            train_dl = DataLoader(dataset_train, batch_size=bs, shuffle=True, generator=RNG)
            val_dl = DataLoader(dataset_val, batch_size=bs, shuffle=False, generator=RNG)
            
            # 1 - Linear Regression Model
            lin_reg = nn.Sequential(
                LinearLayer(2, 2, generator=RNG),
                SoftmaxLayer()
            )
            print(f"Testing linear regression with lr={lr}, bs={bs}")
            
            linear_loss = train(
                train_dl,
                lin_reg,
                CrossEntropyLossLayer(),
                SGDOptimizer(lin_reg.parameters(), lr=lr),
                val_dl,
                epochs
            )
            
            record_best_model("linear regression", lin_reg, 
                              linear_loss["train"], linear_loss["val"], 
                              lr, bs, epochs)
            
            # 2 - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
            sigmoid = nn.Sequential(
                LinearLayer(2, 2, generator=RNG),
                SigmoidLayer(),
                LinearLayer(2, 2, generator=RNG),
                SoftmaxLayer()
            )
            print(f"Testing sigmoid with lr={lr}, bs={bs}")
            
            sigmoid_loss = train(
                train_dl,
                sigmoid,
                CrossEntropyLossLayer(),
                SGDOptimizer(sigmoid.parameters(), lr=lr), 
                val_dl,
                epochs
            )
            
            record_best_model("sigmoid", sigmoid, 
                              sigmoid_loss["train"], sigmoid_loss["val"], 
                              lr, bs, epochs)
            
            # 3 - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
            relu = nn.Sequential(
                LinearLayer(2, 2, generator=RNG),
                ReLULayer(),
                LinearLayer(2, 2, generator=RNG), 
                SoftmaxLayer()
            )
            print(f"Testing relu with lr={lr}, bs={bs}")
            
            relu_loss = train(
                train_dl,
                relu,
                CrossEntropyLossLayer(),
                SGDOptimizer(relu.parameters(), lr=lr),
                val_dl,
                epochs
            )

            record_best_model("relu", relu, 
                              relu_loss["train"], relu_loss["val"], 
                              lr, bs, epochs)
            
            # 4 - Network with two hidden layers (each with size 2)
            #     and Sigmoid, ReLU activation function after corresponding hidden layers
            sigmoid_relu = nn.Sequential(
                LinearLayer(2, 2, generator=RNG),
                SigmoidLayer(),
                LinearLayer(2, 2, generator=RNG),
                ReLULayer(),
                LinearLayer(2, 2, generator=RNG), 
                SoftmaxLayer()
            )
            print(f"Testing sigmoid_relu with lr={lr}, bs={bs}")
            
            sigmoid_relu_loss = train(
                train_dl,
                sigmoid_relu,
                CrossEntropyLossLayer(),
                SGDOptimizer(sigmoid_relu.parameters(), lr=lr),
                val_dl,
                epochs
            )
            
            record_best_model("sigmoid_relu", sigmoid_relu, 
                              sigmoid_relu_loss["train"], sigmoid_relu_loss["val"], 
                              lr, bs, epochs)
            
            # 5 - Network with two hidden layers (each with size 2)
            #     and ReLU, Sigmoid activation function after corresponding hidden layers
            relu_sigmoid = nn.Sequential(
                LinearLayer(2, 2, generator=RNG),
                ReLULayer(),
                LinearLayer(2, 2, generator=RNG),
                SigmoidLayer(),
                LinearLayer(2, 2, generator=RNG), 
                SoftmaxLayer()
            )
            print(f"Testing relu_sigmoid with lr={lr}, bs={bs}")
            
            relu_sigmoid_loss = train(
                train_dl,
                relu_sigmoid,
                CrossEntropyLossLayer(),
                SGDOptimizer(relu_sigmoid.parameters(), lr=lr), 
                val_dl,
                epochs
            )
            
            record_best_model("relu_sigmoid", relu_sigmoid, 
                              relu_sigmoid_loss["train"], relu_sigmoid_loss["val"], 
                              lr, bs, epochs)

            # 6 - Network with two hidden layers (each with size 2)
            #     and ReLU, ReLU activation function after corresponding hidden layers
            relu_2 = nn.Sequential(
                LinearLayer(2, 2, generator=RNG),
                ReLULayer(),
                LinearLayer(2, 2, generator=RNG),
                ReLULayer(),
                LinearLayer(2, 2, generator=RNG), 
                SoftmaxLayer()
            )
            print(f"Testing relu_2 with lr={lr}, bs={bs}")
            
            relu_2_loss = train(
                train_dl,
                relu_2,
                CrossEntropyLossLayer(),
                SGDOptimizer(relu_2.parameters(), lr=lr), 
                val_dl,
                epochs
            )
            
            record_best_model("relu_2", relu_2, 
                              relu_2_loss["train"], relu_2_loss["val"], 
                              lr, bs, epochs)
    
    print("\nBest model configurations:")
    for model_name, config in models.items():
        print(f"{model_name}: lr={config['lr']}, batch_size={config['batch_size']}, min val loss={min(config['val']):.6f}")
    

    for model_name in models:
        models[model_name] = {
            "train": models[model_name]["train"],
            "val": models[model_name]["val"],
            "model": models[model_name]["model"]
        }
    
    return models


@problem.tag("hw3-A")
def accuracy_score(model, dataloader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for CrossEntropy.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is an integer representing a correct class to a corresponding observation.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to MSE accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """
    correct = 0
    total = 0

    with torch.no_grad():
        for observations, targets in dataloader:
            outputs = model.forward(observations)
            _, predicted = torch.max(outputs.data, 1) 

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return correct / total


@problem.tag("hw3-A", start_line=7)
def main():
    """
    Main function of the Crossentropy problem.
    It should:
        1. Call crossentropy_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 10 lines total).
            x-axis should be epochs, y-axis should me Crossentropy loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    dataset_train = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y))
    dataset_val = TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    dataset_test = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))

    ce_configs = crossentropy_parameter_search(dataset_train, dataset_val)

    plt.figure(figsize=(12, 8))
    
    best_model = None
    best_val_loss = float('inf')

    for model, config in ce_configs.items():
        train_loss = config["train"]
        val_loss = config["val"]
        epochs = range(1, len(train_loss) + 1)

        min_val_loss = min(val_loss) 
        if min_val_loss < best_val_loss:
            best_val_loss = min_val_loss
            best_model = config["model"] 
        
        plt.plot(epochs, train_loss, label=f"{model} (train)")
        plt.plot(epochs, val_loss, label=f"{model} (val)")
    
    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"best ce model: {best_model} with loss: {best_val_loss:.6f}")
    plot_model_guesses(DataLoader(dataset_test), best_model, f"Cross Entropy Best Model: {best_model}")
    
    test_accuracy = accuracy_score(best_model, DataLoader(dataset_test))
    print(f"Test accuracy of the best model: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
