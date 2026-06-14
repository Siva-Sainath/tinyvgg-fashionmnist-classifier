# 🧥 TinyVGG FashionMNIST Classifier

A from-scratch PyTorch implementation of the **TinyVGG** convolutional neural network for **10-class fashion item classification** using the FashionMNIST dataset. This project explores the progression from simple linear baselines to CNN-based architectures, demonstrating how convolutional feature extraction outperforms fully connected networks on image data.

---

## 📌 Project Highlights

- ✅ Built and trained **3 model architectures** from scratch in PyTorch — progressing from a linear baseline to a full CNN
- ✅ Implemented custom **training & evaluation loops** with per-epoch loss/accuracy tracking
- ✅ **TinyVGG** (CNN) achieves **88.28% test accuracy**, outperforming linear baselines by up to 13%
- ✅ **Loss curve visualization** and **model comparison benchmarks** across all architectures
- ✅ Model **checkpointing** — trained weights saved and reloadable for inference
- ✅ Follows the [CNN Explainer TinyVGG](https://poloclub.github.io/cnn-explainer/) architecture faithfully

---

## 📊 Model Comparison Results

| Model | Architecture | Test Loss | Test Accuracy |
|---|---|---|---|
| FashionMNISTModelV0 | Flatten → Linear | 0.4766 | 83.43% |
| FashionMNISTModelV1 | Linear + ReLU | 0.6746 | 75.64% |
| **FashionMNISTModelV2** | **TinyVGG (CNN)** ✅ | **0.3232** | **88.28%** |

> TinyVGG outperformed both linear baselines, demonstrating the advantage of convolutional feature extraction for image classification tasks.

---

## 🗂️ Dataset

**FashionMNIST** (via `torchvision.datasets`)

| Property | Value |
|---|---|
| Total images | 70,000 |
| Training set | 60,000 |
| Test set | 10,000 |
| Image size | 28 × 28 pixels (grayscale) |
| Classes | 10 |
| Approx. size | ~30 MB |

**Classes:** T-shirt/Top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot

> FashionMNIST is a widely used computer vision benchmark designed as a more challenging, real-world drop-in replacement for the original MNIST digit dataset.

---

## 🏗️ TinyVGG Architecture

```
Input (1×28×28)
     ↓
Conv Block 1:  Conv2d(1→10, 3×3) → ReLU → Conv2d(10→10, 3×3) → ReLU → MaxPool2d(2×2)
     ↓
Conv Block 2:  Conv2d(10→10, 3×3) → ReLU → Conv2d(10→10, 3×3) → ReLU → MaxPool2d(2×2)
     ↓
Flatten
     ↓
Classifier:   Linear(10×5×5 → 10)
     ↓
Output (10 classes)
```

Based on the [CNN Explainer](https://poloclub.github.io/cnn-explainer/) TinyVGG architecture.

---

## 📁 Repository Structure

```
tinyvgg-fashionmnist-classifier/
│
├── 03_pytorch_computer_vision.ipynb   # Main notebook — all models, training, evaluation, plots
├── helper_functions.py                # Shared utilities: accuracy_fn, plot_loss_curves, etc.
├── models/
│   └── 03_pytorch_computer_vision_model_2.pth   # Saved TinyVGG weights
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision torchmetrics matplotlib
```

### Run the Notebook

```bash
jupyter notebook 03_pytorch_computer_vision.ipynb
```

The notebook will:
1. Download FashionMNIST automatically via `torchvision.datasets`
2. Train all 3 model variants
3. Plot training/test loss curves
4. Compare model performance in a summary table
5. Save the best model checkpoint to `models/`

### Load Saved Model for Inference

```python
import torch
from torch import nn

class FashionMNISTModelV2(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
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

## 🛠️ Tech Stack

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)

- **Framework:** PyTorch
- **Dataset:** `torchvision.datasets.FashionMNIST`
- **Visualization:** Matplotlib
- **Metrics:** Custom `accuracy_fn` (% correct predictions)
- **Environment:** Jupyter Notebook / Google Colab compatible

---

## 📚 Key Concepts Demonstrated

- Building CNN architectures from scratch with `nn.Module`
- Custom `DataLoader` pipelines with `torch.utils.data`
- Training loops with loss backpropagation and optimizer steps
- Evaluation loops with `torch.inference_mode()`
- Loss curve plotting for diagnosing underfitting/overfitting
- Model checkpointing with `torch.save()` and `torch.load()`
- Comparing multiple architectures with a results DataFrame

---

## 🔗 References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [CNN Explainer — TinyVGG](https://poloclub.github.io/cnn-explainer/)
- [FashionMNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [Zero to Mastery: Learn PyTorch for Deep Learning](https://www.learnpytorch.io/)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
