#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include <cstdlib>
#include <ctime>

#define BLOCK_SIZE 16          
#define N 1024

using namespace std;

__global__ void GPU_matMul(const int *matrixA, const int *matrixB, int n, int *matrixC)
{
	int   indexA = n * BLOCK_SIZE * blockIdx.y + n * threadIdx.y;
	int   indexB = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	float sum = 0;
	for (int k = 0; k < n; k++)
		sum += matrixA[indexA + k] * matrixB[indexB + k * n];
	int indexC = n * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE * blockIdx.x;
	matrixC[indexC + n * threadIdx.y + threadIdx.x] = sum;
}

int** CPU_matMul(int** matrixA, int** matrixB, int** matrixC, int n) {
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
		{
			matrixC[i][j] = 0;
			for (int k = 0; k < n; k++)
				matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
		}
	return matrixC;
}

bool checkResult(int** matrixA, int* matrixB, int n) {
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			if (matrixA[i][j] != matrixB[N * i + j])
				return false;
	return true;
}

int main()
{
	int** matrixA_CPU;
	int** matrixB_CPU;
	int** matrixC_CPU;
	matrixA_CPU = new int*[N];
	for (int i = 0; i < N; i++)
		matrixA_CPU[i] = new int[N];

	matrixB_CPU = new int*[N];
	for (int i = 0; i < N; i++)
		matrixB_CPU[i] = new int[N];

	matrixC_CPU = new int*[N];
	for (int i = 0; i < N; i++)
		matrixC_CPU[i] = new int[N];

	int* hostA = new int[N*N];
	int* hostB = new int[N*N];
	int* hostC = new int[N*N];

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			int	k = N * i + j;
			hostA[k] = (rand()) % 10 + 1;
			hostB[k] = (rand()) % 10 + 1;
			matrixA_CPU[i][j] = hostA[k];
			matrixB_CPU[i][j] = hostB[k];
			matrixC_CPU[i][j] = 0;
		}

	}
	clock_t time;
	time = clock();
	matrixC_CPU = CPU_matMul(matrixA_CPU, matrixB_CPU, matrixC_CPU, N);
	time = clock() - time;

	int *deviceA = 0;
	int *deviceB = 0;
	int *deviceC = 0;
	int memory = N * N * sizeof(int);



	cudaError_t cuerr = cudaMalloc((void**)&deviceA, memory);
	if (cuerr != cudaSuccess) {
		fprintf(stderr, "Cannot allocate device array for A: %s\n");
		cudaGetErrorString(cuerr);
	}
	cuerr = cudaMalloc((void**)&deviceB, memory);
	if (cuerr != cudaSuccess) {
		fprintf(stderr, "Cannot allocate device array for B: %s\n");
		cudaGetErrorString(cuerr);
	}
	cuerr = cudaMalloc((void**)&deviceC, memory);
	if (cuerr != cudaSuccess) {
		fprintf(stderr, "Cannot allocate device array for C: %s\n");
		cudaGetErrorString(cuerr);
		return 0;
	}

	//Создание обработчиков событий (определение времени)
	cudaEvent_t start, stop;
	//Инициализация переменной времени расчета
	float gpuTime = 0.0f;

	cuerr = cudaEventCreate(&start);
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot create CUDA start event: %s\n",
			cudaGetErrorString(cuerr));
		return 0;
	}
	cuerr = cudaEventCreate(&stop);
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot create CUDA end event: %s\n",
			cudaGetErrorString(cuerr));
		return 0;
	}

	cudaEventRecord(start, 0);

	//Копирование данных с хоста на девайс
	cuerr = cudaMemcpy(deviceA, hostA, memory, cudaMemcpyHostToDevice);
	if (cuerr != cudaSuccess) {
		fprintf(stderr, "Cannot copy matrix A from host to device: %s\n",
			cudaGetErrorString(cuerr));
		return 0;
	}

	cuerr = cudaMemcpy(deviceB, hostB, memory, cudaMemcpyHostToDevice);
	if (cuerr != cudaSuccess) {
		fprintf(stderr, "Cannot copy B from host to device: %s\n",
			cudaGetErrorString(cuerr));
		return 0;
	}

	//Установка точки старта
	cudaEventRecord(start, 0);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	//Размер сетки 
	dim3 GRID_SIZE(N / threads.x, N / threads.y);
	//Запуск ядра
	GPU_matMul << <GRID_SIZE, threads >> > (deviceA, deviceB, N, deviceC);

	cuerr = cudaGetLastError();
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot launch CUDA kernel GPU_matMul: %s\n",
			cudaGetErrorString(cuerr));
		return 0;
	}

	cuerr = cudaDeviceSynchronize();
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot synchronize CUDA kernel: %s\n",
			cudaGetErrorString(cuerr));
		return 0;
	}

	cuerr = cudaDeviceSynchronize();
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot synchronize CUDA kernel: %s\n",
			cudaGetErrorString(cuerr));
		return 0;
	}

	//Установка точки окончания
	cuerr = cudaEventRecord(stop, 0);
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot stop record CUDA event: %s\n",
			cudaGetErrorString(cuerr));
		return 0;
	}
	//Копирование результата на хост
	cuerr = cudaMemcpy(hostC, deviceC, memory, cudaMemcpyDeviceToHost);
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot copy C from device to host: %s\n",
			cudaGetErrorString(cuerr));
		return 0;
	}

	cudaEventElapsedTime(&gpuTime, start, stop);

	//Проверка правильности результата и расчет времени
	if (checkResult(matrixC_CPU, hostC, N))
	{
		cout << "CPU: " << time * 1000.0 / CLOCKS_PER_SEC << " ms" << endl;
		cout << "GPU: " << gpuTime << " ms";
	}

	//Освобождение памяти 
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);
	free(hostA);
	free(hostB);
	free(hostC);
	return 0;
}

