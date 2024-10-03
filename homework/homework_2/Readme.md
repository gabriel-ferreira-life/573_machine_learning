# Homework 2 - Support Vector Machine 

This repository contains implementations and theoretical derivations involving Support Vector Machine (SVM) models. The focus is on understanding the dual formulation of SVMs and comparing different kernel functions. We thoroughly explored **hard-margin SVM**, **soft-margin SVM**, and validated kernel functions using hand calculations to gain deeper insights into their properties.

## Summary

This homework involved both theoretical and practical aspects of SVMs:

1. **Theoretical Derivations:**
   - Derivation of the dual problem for a toy dataset using hard-margin and soft-margin SVM formulations.
   - Analysis of the positive semi-definiteness and symmetry properties of the kernel function $K(x_i, x_j) = -x_i^T x_j$ using hand-calculated counterexamples.

2. **Practical Implementation:**
   - Implementation of SVM models using different values of the cost parameter $C$ and kernel functions.
   - Evaluation of model performance based on accuracy and the number of support vectors.

### Key Theoretical Results:

- **Dual Problem Formulation:**
  - Derived the dual problem for a toy dataset $\{ ([0,0], -1), ([2,2], -1), ([2,0], +1) \}$ and computed the optimal Lagrange multipliers $\alpha$ through partial derivatives.
  - Demonstrated that a point can lie on the margin boundary $y_i(w^T x_i + w_0) = 1$ without having a positive Lagrange multiplier $(\alpha_i = 0)$, using a specific point from the toy dataset.

- **Kernel Function Validation:**
  - Verified that $K(x_i, x_j) = -x_i^T x_j$ is symmetric but **not a valid kernel function** because it fails the positive semi-definiteness test.
  - Provided a counterexample to show that $\mathbf{z}^T K \mathbf{z}$ can be negative, demonstrating that $K$ does not satisfy Mercer's theorem.

### Key Experimental Results:

- **Effect of Cost Parameter $C$:**
  - Increasing $C$ led to a decrease in the number of support vectors and stabilization of accuracy at higher values.
  - The best accuracy (96.23%) was achieved consistently for $C = 0.1$ and higher, indicating the model was sufficiently regularized.

- **Kernel Function Comparison:**
  - The **RBF kernel** outperformed the **linear** and **polynomial kernels**, achieving the highest accuracy of **96.23%** with the fewest support vectors (90).
  - The linear kernel required more support vectors than the RBF kernel, highlighting its limitations in handling non-linear boundaries.

### Key Highlights:
- **Hand Calculations**: Detailed mathematical derivations of the dual problem, including solving for $\alpha_1, \alpha_2, \alpha_3$, and verifying the boundary conditions for support vectors.
- **Kernel Function Analysis**: Analytical proofs and counterexamples demonstrating the limitations of certain kernel functions.
- **Model Performance Analysis**: Practical evaluation of model behavior under varying $C$-values and kernel functions.

---

## Full Report

For a detailed breakdown of the theoretical derivations, experimental setup, and full results, please check the [Full Report (PDF)](report/Homework2_Report.pdf).
