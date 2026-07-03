# Geometric and Stochastic Analysis of Discontinuities in Sparse Mixture-of-Experts

[![Paper](https://img.shields.io/badge/Paper-ICML_2026-blue.svg)](https://openreview.net/forum?id=AyrNR1ExN5)

This repository is the official implementation of the paper **[Geometric and Stochastic Analysis of Discontinuities in Sparse Mixture-of-Experts](https://openreview.net/forum?id=AyrNR1ExN5)**, accepted at the **Forty-third International Conference on Machine Learning (ICML 2026)**.

**Authors:** Tho Tran Huu, Huu-Tuan Nguyen, Thien-Hai Nguyen, Nhat-Tri Ho, Viet-Hoang Tran, Tho Quan, and Tan Minh Nguyen

---

## 📂 Directory Structure

This repository contains four main modules related to our work. Each subfolder includes its own `README.md` file with detailed instructions. This top-level README provides an overview and explains the relationships between modules.

- **`GMoE/`**
  Contains code and documentation for the **Image Classification Task - DomainBed Benchmark**.
  - *Note:* Depends on utilities from `our_tutel/`.
  - See details in [`GMoE/README.md`](GMoE/README.md).

- **`Language-Glue/`**
  Contains the implementation of **Language-Glue**, a framework for language-related tasks.
  - *Note:* Relies on shared components from `our_tutel/`.
  - See details in [`Language-Glue/README.md`](Language-Glue/README.md).

- **`LanguageModeling/`**
  Contains code for experiments and models related to **Language Modeling**.
  - *Note:* Works independently and does not depend on `our_tutel/`.
  - See details in [`LanguageModeling/README.md`](LanguageModeling/README.md).

- **`our_tutel/`**
  A shared utility library used by `GMoE/` and `Language-Glue/`.
  - This is the **core reusable component** across projects.
  - See details in [`our_tutel/README.md`](our_tutel/README.md).

---

## 🚀 Getting Started

1. **Module Instructions:** Check the `README.md` file inside each subfolder for specific installation and usage instructions.
2. **Environment Setup:** Install the environment requirements as specified for each module.
3. **Core Utility Setup:** For running `GMoE` or `Language-Glue`, ensure that `our_tutel` is properly set up first.

---

## 📝 Citation

If you find our work or this code useful in your research, please consider citing our paper:

```bibtex
@inproceedings{huu2026geometric,
  title={Geometric and Stochastic Analysis of Discontinuities in Sparse Mixture-of-Experts},
  author={Tho Tran Huu and Huu-Tuan Nguyen and Thien-Hai Nguyen and Nhat-Tri Ho and Viet-Hoang Tran and Tho Quan and Tan Minh Nguyen},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
  url={https://openreview.net/forum?id=AyrNR1ExN5}
}
```
# Smooth_SMoE
