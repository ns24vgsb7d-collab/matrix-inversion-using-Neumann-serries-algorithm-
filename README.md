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

## Code Release, Licensing, and Responsibility

This repository is released under the **Apache License 2.0**, a permissive open-source licence that allows anyone to use, modify, and redistribute the code (commercially or non-commercially), as long as proper attribution is maintained.  
I selected Apache 2.0 because it provides:

- explicit permission for reuse and modification,
- strong patent protection,
- clarity for both contributors and users,
- a standard legal framework widely used in scientific and industrial software.

The license also includes a **“no warranty” clause**, which states that the code is provided *“as-is”* without guarantees of correctness, performance, or suitability for any specific use.  
As a result, **I am not legally responsible for any malfunction, damage, or incorrect results** caused by the use of this code.

---

## Responsibilities and Code Maintenance

By releasing this source code publicly, I acknowledge the following responsibilities:

### **1. Transparency**
I provide clear documentation, input/output formats, and instructions enabling others to reproduce my results.

### **2. Limited Liability**
While the code is available for academic and educational use,  
**I cannot guarantee the absence of bugs**, numerical issues, or unexpected behavior on different hardware.

### **3. Bug Handling and Fixes**
If bugs are reported through GitHub Issues:
- I may review the problem,
- propose a fix or workaround,
- or accept pull requests from the community.

However, **I am not obligated** to provide continuous support or long-term maintenance.

### **4. Good Practices for Open-Source Release**
This repository follows recommended GitHub release practices:
- clean commit history,
- removal of unnecessary build artifacts,
- inclusion of a `.gitignore`,
- clear licensing,
- documented algorithm, assumptions, and limits.

---

## Alternatives in Code Release Practices

Different release approaches could have been used:

### **A. Closed-source**
- Code not shared publicly
- Maximum control, but no transparency  
Not appropriate for academic reproducibility.

### **B. Open-source with restrictive license (e.g., GPL)**
- Forces derivative works to remain open-source  
- Strong protections but too restrictive for this context.

### **C. Open-source permissive license (Apache 2.0, MIT)**
- Allows broad reuse
- Compatible with academic projects  
→ **Chosen option**, best balance between openness and low maintenance burden.

### **D. Release only binaries (no source)**
- Protects intellectual property  
- Not suitable for demonstrating understanding of CUDA or algorithmic implementation.

---

## Conclusion on Responsibilities

The chosen licensing and release strategy ensures:
- academic transparency,
- freedom for others to study or extend the code,
- while clearly limiting my legal and maintenance responsibilities.

Users applying this code in their own projects or publications are responsible for validating the correctness of the results and ensuring numerical reliability.


