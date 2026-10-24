# Laboratory of Computational Physics MOD B — CNN Assignment

Group assignment developed with G. Bregolin, L. Fogarolo, and F. Peron (Group 13, Assignment 2: CNNs).

## Overview

This project studies the **generalization capacity of a convolutional neural network (CNN) used as a signal detector**. The dataset simulates a known-shape signal (a localized sinusoidal pattern) buried in random-walk noise, a simplified analogue of problems such as gravitational-wave detection.

The CNN is treated as a **self-learning matched filter**, which allows us to derive an analytical upper limit on the signal-to-noise ratio (SNR) that the network can exploit. We then use the SNR as a single measure of how classifiable a dataset is and how much the CNN can learn from it.

## Objectives

1. Define a measure of the classifiability of a dataset and of the CNN's learning potential over it.
2. Generalize the result to datasets with higher-dimensional feature spaces.
3. Test convergence time and accuracy over datasets with varying SNR.
4. Test the CNN's generalization capacity when trained and tested at different SNRs.
5. Study how regularization affects the learning process.

## Methods

* **Data**: 2-channel time series (N = 10000 samples, length 60) generated as biased random walks, with 3 classes: pure noise, positive sinusoidal pattern, negative sinusoidal pattern.
* **SNR**: the matched-filter SNR is derived analytically (see the Appendix of the report) and compared with the accuracy obtained by the CNN.
* **Statistical test**: a chi-square test checks whether different pairs (amplitude, noise) with the same SNR give the same accuracy distribution. In 2D, a strong hypothesis (same total SNR) and a weak one (same total SNR and same SNR per feature) are tested.
* **Convergence**: epochs needed to reach a target accuracy as a function of the SNR.
* **Cross-SNR generalization**: models trained at one SNR and tested at all the others, shown as heatmaps.
* **Regularization**: L1, L2 and L1L2 with strengths between 1e-6 and 1e-1, comparing validation accuracy on real labels with training accuracy on random labels, following the memorization experiments of Arpit et al.

## Main results

* The SNR is a reliable measure of the dataset's learning potential in 1D. In 2D the strong hypothesis is rejected, but the weak one is not: for a fixed information partition between the two channels, the SNR remains a good measure. The deviation is explained by a **winner-take-all behavior** of the CNN, which relies mainly on the more informative channel.
* Test accuracy grows with the training SNR, but **generalization across SNRs does not**. An intermediate training SNR (around 6) generalizes best.
* For each target accuracy there is an **SNR cutoff** above which the CNN converges in fewer than 50 epochs. This is consistent with the Fisher-information-based spectral thresholds in the literature.
* The effect of regularization on the memorization-generalization trade-off depends on the type of regularizer and on its strength.

## Repository structure

* Notebooks and scripts with the data generation, model training and analyses.
* `report.pdf`: the full report (add the file if you upload it).

## How to run

Install the required dependencies (`numpy`, `tensorflow`/`keras`, `matplotlib`, `scipy`) and run the notebooks using Jupyter.

## References

* D. Arpit et al., *A Closer Look at Memorization in Deep Networks*, arXiv:1706.05394 (2017).
* J. Yan et al., *Generalized Approach to Matched Filtering using Neural Networks*, arXiv:2104.03961 (2022).
* W. H.-C. Huang, *Spectral Thresholds for Identifiability and Stability: Finite-Sample Phase Transitions in High-Dimensional Learning*, arXiv:2510.03809 (2025).
