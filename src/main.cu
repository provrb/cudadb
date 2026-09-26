#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <inttypes.h>
#include <cuda/cmath>
#include <assert.h>
#include <iostream>
#include <iomanip>

// config
constexpr uint8_t MAX_ARRAY_VALUES = 200;

template <typename T>
struct Column {
    size_t colIdx;
    size_t numRows; // len of data
    T* rowCells;    // {data, 
                    //  data, 
                    //  data, 
                    //  data,
                    //  ...}

    Column<T>(size_t colIdx)
        : rowCells(new T[MAX_ARRAY_VALUES]), colIdx(colIdx), numRows(0) {
    }

    // Move all data from this Column into a Column based on device memory. Malloc space for rowCells.
    __host__ Column memMoveToDevice() {
        size_t rowDataSize = sizeof(T) * this->numRows;
        Column<T> gpuColumn = *this;

        cudaMalloc(&gpuColumn.rowCells, rowDataSize);
        cudaMemcpy(
            gpuColumn.rowCells, 
            this->rowCells, 
            rowDataSize,
            cudaMemcpyHostToDevice
        );

        return gpuColumn;
    }

    // Move this rowCells data from GPU memory into hostColumns CPU memory. Free devices rowCells field.
    // In the future, could use a queue. 
    // Push columns to move from gpu to host and then chunk the requests instead of doing one every for-loop
    __host__ void memMoveToHost(Column* __restrict__ hostColumn) {    
        cudaMemcpy(
            hostColumn->rowCells, 
            this->rowCells, 
            sizeof(T) * this->numRows,
            cudaMemcpyDeviceToHost
        );
        cudaFree(this->rowCells);
    }
};

// columnar table version
struct Table {
    void** columns;
    size_t numCols;
    size_t numRows;

    __host__ Table()
        : columns(new void*[MAX_ARRAY_VALUES]), numCols(0), numRows(0) {
    };

    __host__ ~Table() {
        delete[] columns;
    }

    template<typename T>
    __host__ __device__ inline Column<T>* getColumn(const size_t& index) {
        return reinterpret_cast<Column<T>*>(this->columns[index]);
    }
};

template <typename T>
__global__ void DIV(Column<T> col, double divisor) {
    if (divisor == 0) return;
    if (!cuda::std::is_arithmetic<T>::value) {
        printf("error unsupported div operation of data not of arithmetic type.\n");
        return;
    }
    
    uint64_t kernelDatasetIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (kernelDatasetIdx > col.numRows) return;

    *(&col.rowCells[kernelDatasetIdx]) /= divisor;
}

template <typename T>
__global__ void SUM(Column<T> col, int* result) {
    uint64_t kernelDatasetIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (kernelDatasetIdx > col.numRows) {
        return;
    }

    atomicAdd(result, col.rowCells[kernelDatasetIdx]);
}

void printDataset(Table* dataset) {
    for (size_t i = 0; i < dataset->numRows; i++) {
        for (size_t j = 0; j < dataset->numCols; j++) {
            Column<uint64_t>* col = dataset->getColumn<uint64_t>(j);
            std::cout << std::left << std::setw(7) << col->rowCells[j];
        }

        std::cout << std::endl;
    }
}

int main() {
    // malloc dataset on host
    Table* dataset = new Table;
    
    // populate with dummy data
    for (size_t i=0; i<6; i++){
        // malloc column on host
        Column<uint64_t>* col = new Column<uint64_t>(i);
        col->rowCells = new uint64_t[MAX_ARRAY_VALUES];
        for (int j=0; j<20; j++) {
            col->numRows = j+1;
            col->rowCells[j] = 10;
        }
        
        dataset->numRows = std::max(dataset->numRows, col->numRows);
        dataset->columns[i] = col;
        dataset->numCols = i+1;
    }

    printDataset(dataset);
    
    int* hostDatasetSum = new int;
    int* gpuDatasetSum = nullptr;
    cudaMalloc(&gpuDatasetSum, sizeof(int));
    cudaMemset(gpuDatasetSum, 0, sizeof(int));

    // send all the relevant dataset columns from the host memory to the gpu
    for (size_t i = 0; i < dataset->numCols; i++) {
        Column<uint64_t>* hostColumn = dataset->getColumn<uint64_t>(i);
        Column<uint64_t> gpuColumn = hostColumn->memMoveToDevice();

        DIV<<<3, 10>>>(gpuColumn, 10);
        SUM<<<3, 10>>>(gpuColumn, gpuDatasetSum);
        
        gpuColumn.memMoveToHost(hostColumn);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(
        hostDatasetSum,
        gpuDatasetSum,
        sizeof(int),
        cudaMemcpyDeviceToHost
    );

    std::cout << "Sum calculated: " << *hostDatasetSum << std::endl;
    
    printDataset(dataset);

    return 0;
}