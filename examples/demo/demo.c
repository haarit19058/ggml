#include "ggml.h"
#include "ggml-cpu.h"  // For CPU-specific operations
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    // Define the size of our 1D tensors
    const int size = 4;

    // Input data for each tensor
    float data1[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float data2[4] = {10.0f, 20.0f, 30.0f, 40.0f};

    // Calculate an approximate size for the ggml context:
    // - Allocate memory for two tensors (their overhead)
    // - Allocate memory for the tensor data (each tensor: size * sizeof(float))
    // - Allocate extra overhead for the computational graph
    size_t ctx_size = 0;
    ctx_size += 2 * ggml_tensor_overhead();       // Overhead for two tensors
    ctx_size += 2 * (size * sizeof(float));         // Data for two input tensors
    ctx_size += (size * sizeof(float));             // Data for the result tensor
    ctx_size += ggml_graph_overhead();              // Graph overhead
    ctx_size += 1024;                               // Additional overhead

    // Initialize ggml context with the computed memory size
    struct ggml_init_params params = {
        .mem_size   = ctx_size,
        .mem_buffer = NULL,   // Let ggml allocate its own memory
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (ctx == NULL) {
        fprintf(stderr, "Failed to initialize ggml context\n");
        return 1;
    }

    // Create two 1D tensors with 'size' elements of type F32
    struct ggml_tensor * tensor1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size);
    struct ggml_tensor * tensor2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size);

    // Copy input data into the tensor's memory
    memcpy(tensor1->data, data1, size * sizeof(float));
    memcpy(tensor2->data, data2, size * sizeof(float));

    // Build a computation graph
    struct ggml_cgraph * cgraph = ggml_new_graph(ctx);

    // Use ggml_add to perform element-wise addition of tensor1 and tensor2
    struct ggml_tensor * result = ggml_add(ctx, tensor1, tensor2);

    // Mark the 'result' tensor for computation
    ggml_build_forward_expand(cgraph, result);

    // Compute the graph (using 1 thread here)
    ggml_graph_compute_with_ctx(ctx, cgraph, 1);

    // Print the result
    float * res_data = (float *) result->data;
    printf("Result of tensor addition:\n");
    for (int i = 0; i < size; i++) {
        printf("%.2f ", res_data[i]);
    }
    printf("\n");

    // Free the ggml context to release memory
    ggml_free(ctx);
    return 0;
}
