# Understanding and Detecting Adversarial Behavior in Neural Networks

This repository contains the code and experimental framework for an ongoing research project that investigates **adversarial behavior in neural networks**.
The project aims to explore how neural architectures react to adversarial perturbations, to better understand the mechanisms behind their vulnerabilities, and to investigate whether such perturbations can be **detected or explained** effectively. This work is based on several paper and open-source framework referenced at the end of this file.


## Quick Overview

1. **Implement and analyze** a ResNet architecture from scratch using PyTorch.
2. **Generate adversarial examples** using standard attack methods such as FGSM, DeepFool and C&W.
3. **Study representation changes** across layers under adversarial perturbations.
4. **Investigate detection strategies** leveraging activation-space analysis and explainability frameworks.
5. The **metrics** used are classificatin accuracy and cosine similarity between activation tensor.


All experiments are conducted on the CIFAR10 dataset and detailed results will be added progressively as experimentation progresses.


## Repo Structure

```
├── data/              # Data importation
├── model/             # Custom ResNet and architecture modules
├── attacks/           # Adversarial examples used
├── utils/             # Helper utilities for data, plotting, etc.
├── main.ipynb         # Jupyter notebook used for exploration
├── requirements.txt   # Dependencies used
└── README.md          # This document
```

<!-- --- -->

<!-- ## How to Run

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
python train.py --config configs/resnet_baseline.yaml
``` -->

---

## Citation

If you use or refer to this repository, please cite:

> **MANUEL** (2025). *Understanding and Detecting Adversarial Behavior in Neural Networks*. GitHub repository: [[https://github.com/](https://github.com/maegonz/Adversarial-Study)]


## License

This repository is licensed under the **GNU License** (see [LICENSE](./LICENSE) for details).