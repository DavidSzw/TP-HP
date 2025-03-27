#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
//#include <cuda_runtime.h>

// __host__ indique que la fonction est exécutée sur le CPU
// __device__ indique que la fonction est exécutée sur le GPU
__host__ __device__ double gaussian(double x, double sigma) {
    return exp(-(x * x) / (2.0 * sigma * sigma));
}


// __global__ indique que la fonction est un kernel
__global__ void bilateral_filter_kernel(unsigned char *src, unsigned char *dst, int width, int height, int channels, int d, double *spatial_weights, double sigma_color, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        double weight_sum[3] = {0.0, 0.0, 0.0};
        double filtered_value[3] = {0.0, 0.0, 0.0};

        unsigned char *center_pixel = src + (y * width + x) * channels;

        // Itération sur les pixels voisins
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

// fonction main
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

    int d = 5;  // Taille du voisinage
    double sigma_color = 75.0;
    double sigma_space = 75.0;
    int radius = d / 2;

    // Création des poids spatiaux
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

    // Allouer de la mémoire sur le GPU
    unsigned char *d_src, *d_dst;
    double *d_spatial_weights;
    cudaMalloc((void **)&d_src, width * height * channels);
    cudaMalloc((void **)&d_dst, width * height * channels);
    cudaMalloc((void **)&d_spatial_weights, d * d * sizeof(double));

    // copier les données sur le GPU
    cudaMemcpy(d_src, image, width * height * channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_spatial_weights, spatial_weights, d * d * sizeof(double), cudaMemcpyHostToDevice);

    // Définir la taille des blocs et des grilles pour le kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // demmarer le timer
    clock_t start_time = clock();

    // Appeler le kernel pour effectuer le filtrage bilatéral
    bilateral_filter_kernel<<<gridSize, blockSize>>>(d_src, d_dst, width, height, channels, d, d_spatial_weights, sigma_color, radius);

    // attendre la fin de l'exécution du kernel
    cudaDeviceSynchronize();

    // termine le timer
    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Bilateral filter took %.3f seconds\n", elapsed_time);

    // copie des données du GPU vers le CPU
    cudaMemcpy(filtered_image, d_dst, width * height * channels, cudaMemcpyDeviceToHost);

    // sauvegarde de l'image filtrée
    if (!stbi_write_png(argv[2], width, height, channels, filtered_image, width * channels)) {
        printf("Error saving the image!\n");
    }

    // Libérer la mémoire
    free(spatial_weights);
    stbi_image_free(image);
    free(filtered_image);

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_spatial_weights);

    printf("Bilateral filtering complete. Output saved as %s\n", argv[2]);
    return 0;
}