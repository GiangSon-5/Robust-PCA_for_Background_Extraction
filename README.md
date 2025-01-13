# Robust PCA for Background Extraction

# I am using Kaggle's GPU for my project, and here is the link to my project (If you can't access it, it's because I set it to private mode): 
[Kaggle Notebook: Robust-PCA_for_Background_Extraction](https://www.kaggle.com/code/nguyenquyetgiangson/robust-pca-for-background-extraction)

# DEMO
![Demo](https://github.com/GiangSon-5/Robust-PCA_for_Background_Extraction/blob/main/images/demo.jpg)

# Robust PCA

> The classical _Principal Component Analysis_ (PCA) is widely used for high-dimensional analysis and dimensionality reduction. Mathematically, if all the data points are stacked as column vectors of a (n, m)matrix $M$, PCA tries to decompose $M$ as
> 
> $$M = L + S$$
> 
> where $L$ is a rank $k$ ($k<\min(n,m)$) matrix and $S$ is some perturbation/noise matrix. To obtain $L$, PCA solves the following optimization problem
> 
> $$\min_{L} ||M-L||_2$$
> 
> given that rank($L$) <= $k$. However, the effectiveness of PCA relies on the assumption of the noise matrix $S$: $s_{i,j}$ is small and i.i.d. Gaussian. That means PCA is not robust to outliers in data $M$.
> 
> To resolve this issue, Candes, Emmanuel J. et al proposed _Robust Principal Component Analysis_ (Robust PCA or RPCA). The objective is still trying to decompose $M$ into $L$ and $S$, but instead optimizing the following problem
> 
>  <img src="https://github.com/GiangSon-5/Robust-PCA_for_Background_Extraction/blob/main/images/equation.jpg" />
> 
> subject to $L+S = M$.
> 
> Minimizing the $l_1$-norm of $S$ is known to favour sparsity while minimizing the nuclear norm of $L$ is known to favour low-rank matrices (sparsity of singular values). In this way, $M$ is decomposed to a low-rank matrix but not sparse $L$ and a sparse but not low rank $S$. Here $S$ can be viewed as a sparse noise matrix. Robust PCA allows the separation of sparse but outlying values from the original data.
> 
> In addition, Zhou et al. further proposed a "stable" version of Robust PCA, which is called _Stable Principal Component Pursuit_ (Stable PCP or SPCP), which allows a non-sparse Gaussian noise term $Z$ in addition to $L$ and $S$:
> 
> $$M = L+S+Z.$$
> 
> Stable PCP is intuitively more practical since it combines the strength of classical PCA and Robust PCA. However, depending on the exact problem, the proper method should be selected.

There are many [applications of Robust PCA](https://www.comp.nus.edu.sg/~leowwk/cs6101/AY2012-13%20Sem%201/rpca/slides.pdf). Here, we show a few examples of its applications.


## Introduction

This project implements Robust Principal Component Analysis (RPCA) to extract the background from video sequences. RPCA is a powerful technique for decomposing a matrix (in this case, a sequence of video frames) into two components: a low-rank component representing the static background and a sparse component capturing moving objects, foreground details, and noise. This decomposition is particularly useful for background subtraction, a fundamental task in computer vision with applications in surveillance, object tracking, and motion analysis.

Traditional PCA is sensitive to outliers and noise. RPCA addresses this limitation by explicitly modeling the sparse component, making it more robust to corrupted data. By isolating the low-rank background, we can effectively identify and segment moving objects or changes in the scene. This project leverages PyTorch and GPU acceleration to efficiently process video data and perform the RPCA decomposition.

# SSteps Taken in the Project:

## Video Loading and Preprocessing:

- Load the video using OpenCV (cv2).
- Extract frames and store them in a NumPy array.
- Optional: Reduce the video resolution to improve processing speed.

## Data Reshaping:

- Convert the video data into a 2D matrix.
- Convert the NumPy array to a PyTorch tensor for GPU usage.

## RPCA Implementation (RPCA_gpu Class):

- Implement the RPCA_gpu class using PyTorch to perform RPCA on the GPU.
- This class includes methods for p-norm calculation, soft thresholding, SVD-based thresholding, and the iterative RPCA algorithm.

## RPCA Decomposition:

- Create an instance of the RPCA_gpu class with the preprocessed video data.
- Call the fit method to decompose into low-rank (L) and sparse (S) components.

## Visualization:

- Use the imshow_LS function to display the original frame and the extracted background.

## Tools and Libraries:

- **Python**: The main programming language.
- **OpenCV (cv2)**: For video loading and frame extraction.
- **NumPy**: For numerical computation and array manipulation.
- **PyTorch**: For tensor operations, GPU acceleration, and RPCA implementation.
- **matplotlib.pyplot**: For result visualization.
- **imageio**: For image reading (initially imported).
