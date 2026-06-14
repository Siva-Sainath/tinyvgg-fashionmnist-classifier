# TinyVGG FashionMNIST Classifier

A PyTorch implementation of the TinyVGG convolutional neural network trained on the FashionMNIST dataset for 10-class clothing classification. The project progresses through three model architectures — from a simple linear baseline to a CNN — to show how convolutional layers improve performance on image data.

---

## Results

| Model | Architecture | Test Loss | Test Accuracy |
|---|---|---|---|
| FashionMNISTModelV0 | Flatten + Linear | 0.4766 | 83.43% |
| FashionMNISTModelV1 | Linear + ReLU | 0.6746 | 75.64% |
| FashionMNISTModelV2 (TinyVGG) | CNN | 0.3232 | 88.28% |

The TinyVGG CNN outperformed both linear models, confirming that convolutional feature extraction is better suited for image classification than fully connected layers alone.

---

## Dataset

FashionMNIST via `torchvision.datasets` — 70,000 grayscale images at 28x28 pixels (60K train / 10K test), across 10 clothing categories: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot. Approximately 30 MB.

---

## Model Architecture (TinyVGG)

```
Input (1 x 28 x 28)
     |
Conv Block 1:  Conv2d(1->10, 3x3) -> ReLU -> Conv2d(10->10, 3x3) -> ReLU -> MaxPool2d(2x2)
     |
Conv Block 2:  Conv2d(10->10, 3x3) -> ReLU -> Conv2d(10->10, 3x3) -> ReLU -> MaxPool2d(2x2)
     |
Flatten -> Linear(250 -> 10)
     |
Output (10 classes)
```

Based on the [CNN Explainer TinyVGG](https://poloclub.github.io/cnn-explainer/) architecture.

---

## Project Structure

```
tinyvgg-fashionmnist-classifier/
├── 03_pytorch_computer_vision.ipynb   # Main notebook
├── helper_functions.py                # Utilities: accuracy_fn, plot_loss_curves, etc.
├── models/
│   └── 03_pytorch_computer_vision_model_2.pth   # Saved TinyVGG weights
└── README.md
```

---

## Getting Started

```bash
pip install torch torchvision matplotlib
jupyter notebook 03_pytorch_computer_vision.ipynb
```

The notebook downloads FashionMNIST automatically, trains all three models, plots training/test loss curves, and saves the final checkpoint.

### Load the saved model

```python
import torch
from torch import nn

class FashionMNISTModelV2(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 5 * 5, output_shape)
        )

    def forward(self, x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

model = FashionMNISTModelV2(input_shape=1, hidden_units=10, output_shape=10)
model.load_state_dict(torch.load("models/03_pytorch_computer_vision_model_2.pth"))
model.eval()
```

---

## Stack

- PyTorch
- torchvision
- Matplotlib
- Jupyter Notebook

---

## References

- [CNN Explainer — TinyVGG](https://poloclub.github.io/cnn-explainer/)
- [FashionMNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [Zero to Mastery: Learn PyTorch for Deep Learning](https://www.learnpytorch.io/)
