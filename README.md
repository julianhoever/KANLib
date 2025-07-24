# KANLib

This repository contains PyTorch implementations of various neural network modules for Kolmogorov-Arnold networks (KANs). It should simplify development and experimentation with KANs in different domains. The implementation is designed to be modular and extensible, allowing for the easy integration of new features and models.

## Features

- **Supported basis functions:**
    - B-splines
    - Gaussian radial basis functions (RBF)
- **Neural Network Modules:**
    - Linear
- **Other features:**
    - **Grid refinement:** Refine the grid size of the basis functions during training to improve the model's performance (for each layer independently).
    - **Plot learned splines:** Visualize the learned basis functions to understand the model's behavior.
    - **Predefined training loop:** Training loop for common tasks such as regression and classification

## Installation

You can install KANLib via pip:

```bash
pip install git+https://github.com/julianhoever/KANLib.git
```

## KANLib Structure

- `kanlib`
    - `nn`: Contains the neural network modules that can be used to build KANs.
        - `bspline`: Contains all modules related to B-spline basis functions.
        - `gaussian_rbf`: Contains all modules related to Gaussian radial basis functions.
    - `spline_plotting`: Contains functions to visualize the learned basis functions.

## Credits

This project is based on essential works in the field of Kolmogorov-Arnold networks and may use or build on parts of their implementations:

- **KANs: Kolmogorov-Arnold Networks ([Paper](https://arxiv.org/abs/2404.19756) | [Implementation](https://github.com/KindXiaoming/pykan))**
    - Original paper and implementation introducing KANs.
- **Kolmogorov-Arnold Networks are Radial Basis Function Networks ([Paper](https://arxiv.org/abs/2405.06721) | [Implementation](https://github.com/ZiyaoLi/fast-kan))**
    - Implementation of KANs using gaussian radial basis functions.
- **Efficient-KAN: An Efficient Implementation of Kolmogorov-Arnold Network ([Implementation](https://github.com/Blealtan/efficient-kan))**
    - An implementation of KANs with B-splines that focuses on performance improvements.