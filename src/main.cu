#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda/__cmath/ceil_div.h>
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

struct Table {
    size_t* rowIdx;    // {1,    2,    3,      4,    ...}
    size_t* colIdx;    // {1,    2,    3,      4,    ...}
    TYPE_ID* typeIds;  // {INT,  INT,  STRING, CHAR, ...}
    TableValue* data;  // {data, data, data,   data, ...}
};

__global__ void SUM(Table* dataset, size_t datalen, double* sum) {
    uint64_t kernelDatasetIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (kernelDatasetIdx >= datalen)
        return;

    size_t rowIdx = dataset->rowIdx[kernelDatasetIdx];
    size_t colIdx = dataset->colIdx[kernelDatasetIdx];
    TYPE_ID typeId = dataset->typeIds[kernelDatasetIdx];
    TableValue data = dataset->data[kernelDatasetIdx];
    
    switch (typeId) {
        case FLOAT: {
            atomicAdd(sum, data.as_float);
            break;
        }
        case INT: {
            atomicAdd(sum, data.as_int);
            break;
        }
        default:
            printf("unsupported type for SUM operation.\n");
            break;
    }
}

__global__ void DIV(Table* dataset, size_t datalen, size_t divisor) {
    if (divisor == 0) // div by 0
        return;

    uint64_t kernelDatasetIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (kernelDatasetIdx >= datalen)
        return;

    TYPE_ID typeId = dataset->typeIds[kernelDatasetIdx];
    TableValue* data = &dataset->data[kernelDatasetIdx];
    
    switch (typeId) {
        case FLOAT: {
            break;
        }
        case INT: {
            data->as_int /= divisor;
            break;
        }
        default:
            printf("unsupported type for SUM operation.\n");
            break;
    }
}

int main() {
    Table* dataset = nullptr;
    size_t datalen = 0;
    
    cudaMallocManaged(&dataset, sizeof(Table));
    cudaMallocManaged(&dataset->rowIdx, 1000* sizeof(size_t));
    cudaMallocManaged(&dataset->colIdx, 1000* sizeof(size_t));
    cudaMallocManaged(&dataset->typeIds, 1000* sizeof(TYPE_ID));
    cudaMallocManaged(&dataset->data, 2048*sizeof(uint64_t));
    
    // populate with dummy data
    for (size_t i=0; i<10; i++){
        dataset->rowIdx[i] = i+1;
        dataset->colIdx[i] = i+1;
        dataset->typeIds[i] = TYPE_ID::INT;
        dataset->data[i].as_int = 5;

        datalen += 1;
    }

    size_t threads = 256;
    size_t blocks = cuda::ceil_div(datalen, threads);
    
    // double* sum = nullptr;
    // cudaMallocManaged(&sum, sizeof(double));
    
    // SUM<<<blocks, threads>>>(dataset, datalen, sum);
    DIV<<<blocks, threads>>>(dataset, datalen, 5);
    cudaDeviceSynchronize();

    for (size_t i=0; i<datalen; i++) {
        size_t rowIdx = dataset->rowIdx[i];
        size_t colIdx = dataset->colIdx[i];
        TYPE_ID typeId = dataset->typeIds[i];
        TableValue data = dataset->data[i];

        printf("i[%lld, tid: %d] = ", i, typeId);

        switch (typeId) {
            case FLOAT: {
                // double as_dbl = reinterpret_cast<double>(data);
                // printf("%f\n", as_dbl);
                break;
            }
            case INT: {
                printf("%lld\n", data.as_int);
                break;
            }
            default:
                printf("unsupported type for SUM operation.\n");
        }
    }

    // printf("Sum is %f\n", *sum);

    cudaFree(dataset);
    cudaFree(dataset->rowIdx);
    cudaFree(dataset->colIdx);
    cudaFree(dataset->typeIds);
    cudaFree(dataset->data);
    // cudaFree(sum);

    return 0;
}