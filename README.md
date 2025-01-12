# Robust PCA for Background Extraction

## Introduction

This project implements Robust Principal Component Analysis (RPCA) to extract the background from video sequences. RPCA is a powerful technique for decomposing a matrix (in this case, a sequence of video frames) into two components: a low-rank component representing the static background and a sparse component capturing moving objects, foreground details, and noise. This decomposition is particularly useful for background subtraction, a fundamental task in computer vision with applications in surveillance, object tracking, and motion analysis.

Traditional PCA is sensitive to outliers and noise. RPCA addresses this limitation by explicitly modeling the sparse component, making it more robust to corrupted data. By isolating the low-rank background, we can effectively identify and segment moving objects or changes in the scene. This project leverages PyTorch and GPU acceleration to efficiently process video data and perform the RPCA decomposition.
