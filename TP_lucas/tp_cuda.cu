#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>

// Gaussian function
__device__ double gaussian(double x, double sigma) {
    return exp(-(x * x) / (2.0 * sigma * sigma));
}

// CUDA kernel for bilateral filter
__global__ void bilateral_filter_kernel(unsigned char *src, unsigned char *dst, int width, int height, int channels, int d, double sigma_color, double sigma_space, double *spatial_weights) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = d / 2;

    if (x >= radius && x < width - radius && y >= radius && y < height - radius) {
        double weight_sum[3] = {0.0, 0.0, 0.0};
        double filtered_value[3] = {0.0, 0.0, 0.0};

        unsigned char *center_pixel = src + (y * width + x) * channels;

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
            output_pixel[c] = (unsigned char)(filtered_value[c] / (weight_sum[c] + 1e-6));
        }
    }
}

void bilateral_filter(unsigned char *src, unsigned char *dst, int width, int height, int channels, int d, double sigma_color, double sigma_space) {
    int radius = d / 2;

    double *spatial_weights = (double *)malloc(d * d * sizeof(double));
    if (!spatial_weights) {
        printf("Memory allocation for spatial weights failed!\n");
        return;
    }

    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            int x = i - radius, y = j - radius;
            spatial_weights[i * d + j] = gaussian(sqrt(x * x + y * y), sigma_space);
        }
    }

    unsigned char *d_src, *d_dst;
    double *d_spatial_weights;
    size_t img_size = width * height * channels * sizeof(unsigned char);
    size_t weights_size = d * d * sizeof(double);

    cudaMalloc(&d_src, img_size);
    cudaMalloc(&d_dst, img_size);
    cudaMalloc(&d_spatial_weights, weights_size);

    cudaMemcpy(d_src, src, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_spatial_weights, spatial_weights, weights_size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    bilateral_filter_kernel<<<gridDim, blockDim>>>(d_src, d_dst, width, height, channels, d, sigma_color, sigma_space, d_spatial_weights);

    cudaMemcpy(dst, d_dst, img_size, cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_spatial_weights);
    free(spatial_weights);
}

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
    
    bilateral_filter(image, filtered_image, width, height, channels, 5, 75.0, 75.0);

    if (!stbi_write_png(argv[2], width, height, channels, filtered_image, width * channels)) {
        printf("Error saving the image!\n");
        free(filtered_image);
        stbi_image_free(image);
        return 1;
    }

    stbi_image_free(image);
    free(filtered_image);

    printf("Bilateral filtering complete. Output saved as %s\n", argv[2]);
    return 0;
}