# Homework 3 - Decision Tree Learning

This repository contains implementations and theoretical explorations involving Decision Tree Learning. The focus is on constructing decision trees, applying pruning techniques, and analyzing their impact on model performance. Both theoretical derivations and practical experiments are included to gain insights into decision tree behavior, especially with and without pruning.

## Summary

This homework covered both theoretical and practical aspects of decision trees:

1. **Theoretical Exploration:**
   - Conversion of decision rules into equivalent decision tree structures, ensuring the resulting tree representation is consistent with a given rule set.
   - Application of **Jensen’s inequality** to derive properties of decision tree components and understand the computational shortcuts involved in distance calculations in high-dimensional spaces.

2. **Practical Implementation:**
   - Construction of decision trees from training data and implementation of **pruning** techniques to simplify the tree.
   - Evaluation of pruned versus unpruned trees using validation and testing datasets to understand the effect of pruning on model accuracy and complexity.

### Key Theoretical Results:

- **Rule Conversion to Decision Trees:**
  - Demonstrated that any complete and valid set of rules can be converted into an equivalent decision tree, using specific examples to illustrate this construction.
  - Provided a decision tree diagram based on a given set of rules involving attributes `A` and `B` with three classification outcomes.

- **Use of Jensen's Inequality:**
  - Used Jensen’s inequality to derive a **lower bound** for the Euclidean distance between points in high-dimensional space.
  - Showed how this property can be used to make nearest neighbor searches more efficient by pruning the search space, reducing computational cost.

### Key Experimental Results:

- **Pruning Effect on Accuracy:**
  - Pruning the decision tree resulted in decreased accuracy for both validation and test datasets. Validation accuracy dropped from **0.8881** to **0.6895**, while test accuracy decreased from **0.8757** to **0.6908**, suggesting potential underfitting due to pruning.

- **Tree Complexity:**
  - The pruned tree had a significantly **reduced size** (from **288** nodes to **4** nodes) and **depth** (from **7** to **2**). This simplification highlights the trade-off between model complexity and accuracy.

### Key Highlights:
- **Rule-Based Tree Construction**: Converted rule sets into a decision tree structure, illustrating the practical applicability of decision tree learning in converting human-readable rules into a model.
- **Pruning Analysis**: Explored the impact of pruning, noting that while pruning reduced the complexity of the model, it also led to **underfitting**, as evidenced by the drop in accuracy.
- **Efficiency Techniques**: Leveraged properties derived from Jensen's inequality to discuss ways to optimize high-dimensional nearest neighbor searches.

---

## Full Report

For a detailed breakdown of the theoretical derivations, experimental setup, and full results, please check the [Full Report (PDF)](report/Homework3_Report.pdf).
