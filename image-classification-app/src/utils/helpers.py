# helpers.py

def set_seed(seed):
    """Set the random seed for reproducibility."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_model(model, path):
    """Save the trained model to the specified path."""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Load the model weights from the specified path."""
    model.load_state_dict(torch.load(path))
    model.eval()

def visualize_samples(images, labels, class_names):
    """Visualize a batch of images and their corresponding labels."""
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(12, 6))
    for i in range(len(images)):
        ax = plt.subplot(2, 5, i + 1)
        plt.imshow(np.transpose(images[i], (1, 2, 0)))
        plt.title(class_names[labels[i]])
        plt.axis('off')
    plt.show()