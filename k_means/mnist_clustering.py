if __name__ == "__main__":
    from k_means import lloyd_algorithm  # type: ignore
else:
    from .k_means import lloyd_algorithm

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw4-A", start_line=1)
def main():
    """Main function of k-means problem

    Run Lloyd's Algorithm for k=10, and report 10 centers returned.

    NOTE: This code might take a while to run. For debugging purposes you might want to change:
        x_train to x_train[:10000]. Make sure to change it back before submission!
    """
    (x_train, _), _ = load_dataset("mnist")
    centers, errors = lloyd_algorithm(x_train, 10)

    fig, axes = plt.subplots(2, 5)
    axes = axes.flatten()

    for i in range(10):
        axes[i].imshow(centers[i].reshape(28, 28), cmap="gray")
    
    plt.show()



if __name__ == "__main__":
    main()
