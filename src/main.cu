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

// struct of arrays
// instead of arrays of structs. e.g.: aos={{x,y,z}, {a,b,c}}

enum TYPE_ID {
    INT = 1,
    STRING,
    CHAR,
    FLOAT
};

union TableValue {
    uint64_t as_int;
    double as_float;
    void* as_ptr;
};

// all columns have the same type of data
struct Column {
    TYPE_ID typeId;
    size_t colIdx;
    size_t numRows;
    TableValue* data; // {data, 
                      //  data, 
                      //  data, 
                      //  data,
                      //  ...}
};

// columnar table version
struct Table {
    Column** columns;
    size_t numCols;
    size_t numRows;
};

__global__ void DIV(Column col, double divisor) {
    uint64_t kernelDatasetIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (kernelDatasetIdx > col.numRows) {
        return;
    }
    
    if (divisor == 0) {
        printf("error in div: divisor provided in 0. cannot divide by 0\n");
        return;
    }
    
    if (col.typeId != TYPE_ID::INT && col.typeId != TYPE_ID::FLOAT) {
        printf("error unsupported div operation of data not of type INT or FLOAT.\n");
        return;
    }

    TableValue* tableValue = &col.data[kernelDatasetIdx];
    if (tableValue->as_ptr == nullptr) {
        return;
    }

    tableValue->as_int /= divisor;
}

__global__ void SUM(Column col, int* result) {
    uint64_t kernelDatasetIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (kernelDatasetIdx > col.numRows) {
        return;
    }
    printf("%llu + %d\n", col.data[kernelDatasetIdx].as_int, *result);
    atomicAdd(result, col.data[kernelDatasetIdx].as_int);
}

void printDataset(Table* dataset) {
    for (size_t i = 0; i < dataset->numRows; i++) {
        for (size_t j = 0; j < dataset->numCols; j++) {
            Column* col = dataset->columns[j];

            switch (col->typeId) {
                case TYPE_ID::INT:
                    printf("%-7lld", col->data[i].as_int);
                    break;
                case TYPE_ID::FLOAT:
                    printf("%-7f", col->data[i].as_float);
                    break;
                default:
                    printf("%-7s", "UNIMPLEMENTED");
                    break;
            }
        }

        printf("\n");
    }
}

int main() {
    const uint32_t COLUMN_ARRAY_SIZE = 100 * sizeof(Column*);
    const uint32_t DATASET_SIZE = sizeof(Table) + COLUMN_ARRAY_SIZE;
    const uint32_t COLUMN_DATA_SIZE = 100 * sizeof(TableValue);
    const uint32_t COLUMN_SIZE = sizeof(Column) + COLUMN_DATA_SIZE;

    // malloc dataset on host
    Table* dataset = (Table*)malloc(DATASET_SIZE);
    dataset->columns = (Column**)malloc(COLUMN_ARRAY_SIZE);
    dataset->numCols = 0;
    dataset->numRows = 0;
    
    // populate with dummy data
    for (size_t i=0; i<6; i++){
        // malloc column on host
        Column* col = (Column*)malloc(COLUMN_SIZE);
        col->data = (TableValue*)malloc(COLUMN_DATA_SIZE);
        col->typeId = TYPE_ID::INT;
        col->colIdx = i;
        col->numRows = 0;
        
        for (int j=0; j<20; j++) {
            col->numRows = j+1;
            col->data[j].as_int = 10;
            dataset->numRows = col->numRows;
        }
        
        dataset->columns[i] = col;
        dataset->numCols = i+1;
    }

    printDataset(dataset);
    
    int* hostDatasetSum = (int*)malloc(sizeof(int));
    int* gpuDatasetSum = nullptr;
    cudaMalloc(&gpuDatasetSum, sizeof(int));
    cudaMemset(gpuDatasetSum, 0, sizeof(int));

    // send all the relevant dataset columns from the host memory to the gpu
    for (size_t i = 0; i < dataset->numCols; i++) {
        Column* hostColumn = dataset->columns[i];
        size_t rowDataSize = sizeof(TableValue) * hostColumn->numRows;
        Column gpuColumn = *hostColumn;

        cudaMalloc(&gpuColumn.data, rowDataSize);
        cudaMemcpy(
            gpuColumn.data, 
            hostColumn->data, 
            rowDataSize,
            cudaMemcpyHostToDevice
        );

        DIV<<<3, 10>>>(gpuColumn, 10);
        SUM<<<2, 10>>>(gpuColumn, gpuDatasetSum); // get sum of everything in the dataset

        // bring data modified on the gpu back to host
        // perhaps modify the data in DIV in shared memory then 
        // memcpy all the shared memory back into hostcolumns at once
        // rather than in this for loop
        cudaMemcpy(
            hostColumn->data, 
            gpuColumn.data, 
            rowDataSize,
            cudaMemcpyDeviceToHost
        );
        cudaFree(gpuColumn.data);
    }

    cudaDeviceSynchronize();

    cudaMemcpy(
        hostDatasetSum,
        gpuDatasetSum,
        sizeof(int),
        cudaMemcpyDeviceToHost
    );

    printf("sum of everything: %d\n", *hostDatasetSum);

    printDataset(dataset);
    free(dataset->columns);
    free(dataset);

    return 0;
}