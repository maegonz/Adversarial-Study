# Adversarial-Study

## Abstract

This repository contains the code and experimental framework for an ongoing research project that investigates **adversarial behavior in neural networks**.
The project aims to explore how neural architectures react to adversarial perturbations, to better understand the mechanisms behind their vulnerabilities, and to investigate whether such perturbations can be **detected or explained** effectively.

---

## Motivation

Despite their remarkable success, modern neural networks remain fragile in the presence of adversarial inputs: imperceptible perturbations that cause misclassifications with high confidence.
This project focuses on understanding *why* these vulnerabilities occur and explores *whether they can be identified* through internal network activations, concept-based reasoning, or statistical irregularities.

---

## Research Objectives

1. **Implement and analyze** a ResNet architecture from scratch using PyTorch.
2. **Generate adversarial examples** using standard attack methods (FGSM, DeepFool, C&W).
3. **Study representation changes** across layers under adversarial perturbations.
4. **Investigate detection strategies** leveraging activation-space analysis and explainability frameworks.
5. **Connect adversarial robustness and explainability** through shared conceptual structures.

---

## Methodology

* **Framework:** Python, PyTorch, CRAFT, ART
* **Architecture:** Custom ResNet implementation
* **Attack Methods:** FGSM, DeepFool, C&W
* **Datasets:** CIFAR-10
* **Metrics:**

  * Accuracy under adversarial perturbation
  * Feature space distance metrics such as cosine similarity
  * Activation profile statistics

<!-- Each experiment is documented in the `/notebooks` and `/experiments` directories. -->

<!-- ---

## Preliminary Results

Initial findings suggest:

* Adversarial perturbations cause measurable, structured distortions in intermediate activation spaces.
* Certain feature representations remain more robust and interpretable than others.
* Statistical and concept-based detection methods show promise for identifying adversarial inputs before classification.

Comprehensive quantitative results will be added as experimentation progresses. -->

---

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
<!-- 
## Citation

If you use or refer to this repository, please cite:

> **Your Name.** (2025). *Understanding and Detecting Adversarial Behavior in Neural Networks*. GitHub repository: [[https://github.com/](https://github.com/)maegonz/Adversarial-Study]

--- -->

## References

1. T. Fel *et al.*, “CRAFT: Concept Recursive Activation Factorization for Explainability,” *arXiv preprint* arXiv:2211.10154 [cs.CV], 2023.
2. A. C. Serban, E. Poll, and J. Visser, “Adversarial Examples — A Complete Characterisation of the Phenomenon,” *arXiv preprint* arXiv:1810.01185 [cs.CV], 2019.
3. C. Szegedy *et al.*, “Intriguing Properties of Neural Networks,” *arXiv preprint* arXiv:1312.6199 [cs.CV], 2014.
4. A. Shafahi *et al.*, “Are Adversarial Examples Inevitable?,” *arXiv preprint* arXiv:1809.02104 [cs.LG], 2020.
5. N. Carlini and D. Wagner, “Towards Evaluating the Robustness of Neural Networks,” *arXiv preprint* arXiv:1608.04644 [cs.CR], 2017.
6. S. Bessai, M. Martel, and A. Ioualalen, “Online Detection of Adversarial Examples by Activation Profile Inspection,” 2025.
7. E. Poeta *et al.*, “Concept-Based Explainable Artificial Intelligence: A Survey,” *arXiv preprint* arXiv:2312.12936 [cs.AI], 2023.
8. I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and Harnessing Adversarial Examples,” *arXiv preprint* arXiv:1412.6572 [stat.ML], 2015.
9. S.-M. Moosavi-Dezfooli, A. Fawzi, and P. Frossard, “DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks,” *arXiv preprint* arXiv:1511.04599 [cs.LG], 2016.
10. A. Ghorbani *et al.*, “Towards Automatic Concept-Based Explanations,” *arXiv preprint* arXiv:1902.03129 [stat.ML], 2019.
11. R. Zhang *et al.*, “Invertible Concept-Based Explanations for CNN Models with Non-Negative Concept Activation Vectors,” *arXiv preprint* arXiv:2006.15417 [cs.CV], 2021.
12. J. H. Lee *et al.*, “Concept-Based Explanations in Computer Vision: Where Are We and Where Could We Go?,” *arXiv preprint* arXiv:2409.13456 [cs.CV], 2024.

---

## License

This repository is licensed under the **MIT License** (see [LICENSE](./LICENSE) for details).

---

## Acknowledgements

This work builds upon foundational research in adversarial robustness, explainable AI, and neural network interpretability.
Special thanks to the open-source and academic communities whose contributions enable this research.