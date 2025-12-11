#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

#define N 64
#define TILE 16

// Macro pour vérifier les erreurs CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "Erreur CUDA dans " << __FILE__ << ":" << __LINE__ \
                 << " - " << cudaGetErrorString(err) << endl; \
            exit(1); \
        } \
    } while(0)

void loadMatrix(const char* filename, double* M) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Erreur ouverture : " << filename << endl;
        exit(1);
    }
    for (int i = 0; i < N * N; i++) {
        if (!(file >> M[i])) {
            cout << "Erreur lecture matrice" << endl;
            exit(1);
        }
    }
    file.close();
}

void saveMatrix(const char* filename, double* M) {
    ofstream file(filename);
    if (!file.is_open()) {
        cout << "Erreur ouverture : " << filename << endl;
        exit(1);
    }
    file << fixed << setprecision(16);
    for (int i = 0; i < N * N; i++) {
        file << M[i];
        if ((i + 1) % N == 0) file << "\n";
        else file << " ";
    }
    file.close();
}

__global__ void matMul_tiled(const double* A, const double* B, double* C) {
    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    double sum = 0.0;

    for (int t = 0; t < N / TILE; t++) {
        As[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}

// Kernel pour combiner les matrices: C = a*A + b*B + c*D
__global__ void matCombine(const double* A, const double* B, const double* D,
    double* C, double a, double b, double c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) {
        C[idx] = a * A[idx] + b * B[idx] + c * D[idx];
    }
}

// Kernel pour calculer l'erreur relative ||A*Ainv - I||_F
__global__ void computeError(const double* M, double* errors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) {
        int row = idx / N;
        int col = idx % N;
        double expected = (row == col) ? 1.0 : 0.0;
        double diff = M[idx] - expected;
        errors[idx] = diff * diff;
    }
}

void computeNeumann(const char* pathA, const char* pathDinv,
    const char* pathE, const char* pathOutput) {

    cout << "=== Calcul de l'inverse par série de Neumann ===" << endl;

    // Création des événements pour mesurer le temps
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Allocation et chargement des matrices sur CPU
    double* h_A = (double*)malloc(N * N * sizeof(double));
    double* h_Dinv = (double*)malloc(N * N * sizeof(double));
    double* h_E = (double*)malloc(N * N * sizeof(double));

    loadMatrix(pathA, h_A);
    loadMatrix(pathDinv, h_Dinv);
    loadMatrix(pathE, h_E);

    cout << "Matrices chargées (taille: " << N << "x" << N << ")" << endl;

    // Allocation sur GPU
    double* d_A, * d_Dinv, * d_E, * d_X, * d_Y, * d_W, * d_Z, * d_AinvApprox;
    CUDA_CHECK(cudaMalloc(&d_A, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Dinv, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_E, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_X, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Y, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Z, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_AinvApprox, N * N * sizeof(double)));

    // Copie vers GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Dinv, h_Dinv, N * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E, h_E, N * N * sizeof(double), cudaMemcpyHostToDevice));

    dim3 blockDim(TILE, TILE);
    dim3 gridDim(N / TILE, N / TILE);

    CUDA_CHECK(cudaEventRecord(start));

    // Calculs de la série de Neumann
    // X = Dinv * E
    matMul_tiled << <gridDim, blockDim >> > (d_Dinv, d_E, d_X);
    CUDA_CHECK(cudaGetLastError());

    // Y = X * Dinv = Dinv * E * Dinv
    matMul_tiled << <gridDim, blockDim >> > (d_X, d_Dinv, d_Y);
    CUDA_CHECK(cudaGetLastError());

    // W = X * X = (Dinv * E)^2
    matMul_tiled << <gridDim, blockDim >> > (d_X, d_X, d_W);
    CUDA_CHECK(cudaGetLastError());

    // Z = W * Dinv = (Dinv * E)^2 * Dinv
    matMul_tiled << <gridDim, blockDim >> > (d_W, d_Dinv, d_Z);
    CUDA_CHECK(cudaGetLastError());

    // Calcul final: Ainv_approx = Dinv - Y + Z (sur GPU)
    int blockSize = 256;
    int gridSize = (N * N + blockSize - 1) / blockSize;
    matCombine << <gridSize, blockSize >> > (d_Dinv, d_Y, d_Z, d_AinvApprox, 1.0, -1.0, 1.0);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    cout << "Temps de calcul GPU: " << milliseconds << " ms" << endl;

    // Vérification: calcul de A * Ainv
    double* d_verification;
    CUDA_CHECK(cudaMalloc(&d_verification, N * N * sizeof(double)));
    matMul_tiled << <gridDim, blockDim >> > (d_A, d_AinvApprox, d_verification);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Calcul de l'erreur ||A*Ainv - I||_F
    double* d_errors;
    CUDA_CHECK(cudaMalloc(&d_errors, N * N * sizeof(double)));
    computeError << <gridSize, blockSize >> > (d_verification, d_errors);
    CUDA_CHECK(cudaGetLastError());

    double* h_errors = (double*)malloc(N * N * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h_errors, d_errors, N * N * sizeof(double), cudaMemcpyDeviceToHost));

    double errorSum = 0.0;
    for (int i = 0; i < N * N; i++) {
        errorSum += h_errors[i];
    }
    double frobeniusError = sqrt(errorSum);

    cout << fixed << setprecision(10);
    cout << "Erreur de Frobenius ||A*Ainv - I||_F = " << frobeniusError << endl;

    // Copie du résultat vers CPU
    double* h_AinvApprox = (double*)malloc(N * N * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h_AinvApprox, d_AinvApprox, N * N * sizeof(double),
        cudaMemcpyDeviceToHost));

    // Sauvegarde
    saveMatrix(pathOutput, h_AinvApprox);
    cout << "Résultat sauvegardé dans: " << pathOutput << endl;

    // Affichage des premiers éléments
    cout << "\nPremiers éléments de Ainv_approx:" << endl;
    cout << setprecision(16);
    for (int i = 0; i < min(3, N); i++) {
        for (int j = 0; j < min(3, N); j++) {
            cout << h_AinvApprox[i * N + j] << " ";
        }
        cout << endl;
    }

    // Libération de la mémoire
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_Dinv));
    CUDA_CHECK(cudaFree(d_E));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_Z));
    CUDA_CHECK(cudaFree(d_AinvApprox));
    CUDA_CHECK(cudaFree(d_verification));
    CUDA_CHECK(cudaFree(d_errors));

    free(h_A);
    free(h_Dinv);
    free(h_E);
    free(h_AinvApprox);
    free(h_errors);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    const char* pathA = "C:/Users/kabeyad/Desktop/minip2/minip2/A.txt";
    const char* pathDinv = "C:/Users/kabeyad/Desktop/minip2/minip2/Dinv.txt";
    const char* pathE = "C:/Users/kabeyad/Desktop/minip2/minip2/E.txt";
    const char* pathOutput = "C:/Users/kabeyad/Desktop/minip2/minip2/Ainv_cuda.txt";

    computeNeumann(pathA, pathDinv, pathE, pathOutput);

    return 0;
}