#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>

// Gaussian function
__device__ double gaussian(double x, double sigma) {
    return exp(-(x * x) / (2.0 * sigma * sigma));
}

// Kernel for bilateral filter
__global__ void bilateral_filter_kernel(unsigned char *src, unsigned char *dst, int width, int height, int channels, int d, double *spatial_weights, double sigma_color, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        double weight_sum[3] = {0.0, 0.0, 0.0};
        double filtered_value[3] = {0.0, 0.0, 0.0};

        unsigned char *center_pixel = src + (y * width + x) * channels;

        // Iterate over local window
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                int nx = x + j - radius;
                int ny = y + i - radius;

                if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                    continue;
                }

                unsigned char *neighbor_pixel = src + (ny * width + nx) * channels;

                for (int c = 0; c < channels; c++) {
                    double range_weight = gaussian(abs(neighbor_pixel[c] - center_pixel[c]), sigma_color);
                    double weight = spatial_weights[i * d + j] * range_weight;

                    filtered_value[c] += neighbor_pixel[c] * weight;
                    weight_sum[c] += weight;
                }
            }
        }

        unsigned char *output_pixel = dst + (y * width + x) * channels;
        for (int c = 0; c < channels; c++) {
            output_pixel[c] = (unsigned char)(filtered_value[c] / (weight_sum[c] + 1e-6));  // Avoid division by zero
        }
    }
}

// Main function
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input_image> <output_image>\n", argv[0]);
        return 1;
    }

    int width, height, channels;
    unsigned char *image = stbi_load(argv[1], &width, &height, &channels, 0);
    if (!image) {
        printf("Error loading image!\n");
        return 1;
    }

    if (width <= 5 || height <= 5) {
        printf("Image is too small for bilateral filter (at least 5x5 size needed).\n");
        stbi_image_free(image);
        return 1;
    }

    unsigned char *filtered_image = (unsigned char *)malloc(width * height * channels);
    if (!filtered_image) {
        printf("Memory allocation for filtered image failed!\n");
        stbi_image_free(image);
        return 1;
    }

    int d = 5;  // Filter size (e.g., 5x5)
    double sigma_color = 75.0;
    double sigma_space = 75.0;
    int radius = d / 2;

    // Precompute spatial Gaussian weights on the host
    double *spatial_weights = (double *)malloc(d * d * sizeof(double));
    if (!spatial_weights) {
        printf("Memory allocation for spatial weights failed!\n");
        stbi_image_free(image);
        free(filtered_image);
        return 1;
    }

    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            int x = i - radius, y = j - radius;
            spatial_weights[i * d + j] = gaussian(sqrt(x * x + y * y), sigma_space);
        }
    }

    // Allocate memory on GPU
    unsigned char *d_src, *d_dst;
    double *d_spatial_weights;
    cudaMalloc((void **)&d_src, width * height * channels);
    cudaMalloc((void **)&d_dst, width * height * channels);
    cudaMalloc((void **)&d_spatial_weights, d * d * sizeof(double));

    // Copy data to GPU
    cudaMemcpy(d_src, image, width * height * channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_spatial_weights, spatial_weights, d * d * sizeof(double), cudaMemcpyHostToDevice);

    // Set up CUDA kernel execution parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Start timer
    clock_t start_time = clock();

    // Launch the CUDA kernel
    bilateral_filter_kernel<<<gridSize, blockSize>>>(d_src, d_dst, width, height, channels, d, d_spatial_weights, sigma_color, radius);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // End timer
    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Bilateral filter took %.3f seconds\n", elapsed_time);

    // Copy result back to CPU
    cudaMemcpy(filtered_image, d_dst, width * height * channels, cudaMemcpyDeviceToHost);

    // Save the output image
    if (!stbi_write_png(argv[2], width, height, channels, filtered_image, width * channels)) {
        printf("Error saving the image!\n");
    }

    // Free memory
    free(spatial_weights);
    stbi_image_free(image);
    free(filtered_image);

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_spatial_weights);

    printf("Bilateral filtering complete. Output saved as %s\n", argv[2]);
    return 0;
}