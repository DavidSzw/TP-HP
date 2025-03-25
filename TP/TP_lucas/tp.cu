#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Erreur CUDA dans %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Fonction Gaussienne
double gaussian(double x, double sigma) {
    return exp(-(x * x) / (2.0 * sigma * sigma));
}

// Filtre bilatéral sur CPU
void bilateral_filter_cpu(unsigned char *src, unsigned char *dst, int width, int height, int channels, int d, double sigma_color, double sigma_space) {
    int radius = d / 2;

    double *spatial_weights = (double *)malloc(d * d * sizeof(double));
    if (!spatial_weights) {
        printf("Échec de l'allocation mémoire pour les poids spatiaux !\n");
        return;
    }

    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            int x = i - radius, y = j - radius;
            spatial_weights[i * d + j] = gaussian(sqrt(x * x + y * y), sigma_space);
        }
    }

    for (int y = radius; y < height - radius; y++) {
        for (int x = radius; x < width - radius; x++) {
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

    free(spatial_weights);
}

// Noyau CUDA pour le filtre bilatéral
__global__ void bilateral_filter_kernel(
    unsigned char *src, 
    unsigned char *dst, 
    int width, 
    int height, 
    int channels, 
    int d, 
    double *spatial_weights, 
    double sigma_color, 
    double sigma_space) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = d / 2;

    if (x < radius || x >= width - radius || y < radius || y >= height - radius) {
        return;
    }

    double weight_sum[3] = {0.0, 0.0, 0.0};
    double filtered_value[3] = {0.0, 0.0, 0.0};

    int center_idx = (y * width + x) * channels;
    
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            int nx = x + j - radius;
            int ny = y + i - radius;

            int neighbor_idx = (ny * width + nx) * channels;

            for (int c = 0; c < channels; c++) {
                double range_weight = gaussian(abs(src[neighbor_idx + c] - src[center_idx + c]), sigma_color);
                double weight = spatial_weights[i * d + j] * range_weight;

                filtered_value[c] += src[neighbor_idx + c] * weight;
                weight_sum[c] += weight;
            }
        }
    }

    int output_idx = (y * width + x) * channels;
    for (int c = 0; c < channels; c++) {
        dst[output_idx + c] = (unsigned char)(filtered_value[c] / (weight_sum[c] + 1e-6));
    }
}

// Filtre bilatéral sur GPU
void bilateral_filter(unsigned char *src, unsigned char *dst, int width, int height, int channels, int d, double sigma_color, double sigma_space) {
    int radius = d / 2;

    // Précalcul des poids spatiaux sur CPU
    double *spatial_weights = (double *)malloc(d * d * sizeof(double));
    if (!spatial_weights) {
        printf("Échec de l'allocation mémoire pour les poids spatiaux !\n");
        return;
    }
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            int x = i - radius, y = j - radius;
            spatial_weights[i * d + j] = gaussian(sqrt(x * x + y * y), sigma_space);
        }
    }

    // Allocation mémoire sur GPU
    unsigned char *d_src, *d_dst;
    double *d_spatial_weights;

    CUDA_CHECK(cudaMalloc(&d_src, width * height * channels * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_dst, width * height * channels * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_spatial_weights, d * d * sizeof(double)));

    // Copie des données vers GPU
    CUDA_CHECK(cudaMemcpy(d_src, src, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spatial_weights, spatial_weights, d * d * sizeof(double), cudaMemcpyHostToDevice));

    // Configuration de la grille et des blocs
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Lancement du noyau
    bilateral_filter_kernel<<<gridDim, blockDim>>>(d_src, d_dst, width, height, channels, d, d_spatial_weights, sigma_color, sigma_space);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Récupération des résultats
    CUDA_CHECK(cudaMemcpy(dst, d_dst, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // Libération mémoire GPU
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFree(d_spatial_weights));

    free(spatial_weights);
}

// Fonction principale
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage : %s <image_entrée> <image_sortie>\n", argv[0]);
        return 1;
    }

    int width, height, channels;
    unsigned char *image = stbi_load(argv[1], &width, &height, &channels, 0);
    if (!image) {
        printf("Erreur lors du chargement de l'image !\n");
        return 1;
    }

    if (width <= 5 || height <= 5) {
        printf("L'image est trop petite pour le filtre bilatéral (taille minimale 5x5 requise).\n");
        stbi_image_free(image);
        return 1;
    }

    unsigned char *filtered_image = (unsigned char *)malloc(width * height * channels);
    if (!filtered_image) {
        printf("Échec de l'allocation mémoire pour l'image filtrée !\n");
        stbi_image_free(image);
        return 1;
    }
    
    bilateral_filter(image, filtered_image, width, height, channels, 5, 75.0, 75.0);

    if (!stbi_write_png(argv[2], width, height, channels, filtered_image, width * channels)) {
        printf("Erreur lors de la sauvegarde de l'image !\n");
        free(filtered_image);
        stbi_image_free(image);
        return 1;
    }

    stbi_image_free(image);
    free(filtered_image);

    printf("Filtrage bilatéral terminé. Sortie sauvegardée sous %s\n", argv[2]);
    return 0;
}