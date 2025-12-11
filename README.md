# Matrix Inversion Using Neumann Series (CUDA)

This project implements matrix inversion using the **Neumann series algorithm** accelerated with **CUDA**.  
The objective is to approximate the inverse of a 64×64 matrix by exploiting GPU parallelism and optimized tiled matrix multiplication.

The Neumann approximation used is:
\[
A^{-1} \approx D^{-1} - D^{-1} E D^{-1} + (D^{-1}E)^2 D^{-1}
\]
where `D` is the diagonal part of `A` and `E = D - A`.

The implementation includes optimized CUDA kernels, error verification, and matrix I/O for reproducible experiments.

---

## Main Features
- **Tiled CUDA matrix multiplication** (`matMul_tiled`) using shared memory
- **Neumann series inversion** entirely on GPU
- **Custom kernels** for matrix combination and error computation
- **Frobenius norm error**: evaluates accuracy of the approximate inverse
- **CUDA event timing** to measure GPU performance
- **Matrix loading/saving** from `.txt` files

---

## File Structure
