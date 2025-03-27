#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

// Gaussian function
double gaussian(double x, double sigma) {
    return exp(-(x * x) / (2.0 * sigma * sigma));
}

// Filtre bilatéral manuel
void bilateral_filter(unsigned char *src, unsigned char *dst, int width, int height, int channels, int d, double sigma_color, double sigma_space) {
    int radius = d / 2;

    // Pré-calculer les poids gaussiens spatiaux
    double *spatial_weights = (double *)malloc(d * d * sizeof(double));
    if (!spatial_weights) {
        printf("Échec de l'allocation de mémoire pour les poids spatiaux!\n");
        return;
    }

    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            int x = i - radius, y = j - radius;
            spatial_weights[i * d + j] = gaussian(sqrt(x * x + y * y), sigma_space);
        }
    }

    // Traiter l'image
    for (int y = radius; y < height - radius; y++) {
        for (int x = radius; x < width - radius; x++) {
            double weight_sum[3] = {0.0, 0.0, 0.0};
            double filtered_value[3] = {0.0, 0.0, 0.0};

            // Obtenir le pointeur du pixel central
            unsigned char *center_pixel = src + (y * width + x) * channels;

            // Itérer sur la fenêtre locale
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    int nx = x + j - radius;
                    int ny = y + i - radius;

                    // Vérification des limites pour s'assurer que nous sommes dans l'image
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                        continue;
                    }

                    // Obtenir le pointeur du pixel voisin
                    unsigned char *neighbor_pixel = src + (ny * width + nx) * channels;

                    for (int c = 0; c < channels; c++) {
                        // Calculer le poids de la plage
                        double range_weight = gaussian(abs(neighbor_pixel[c] - center_pixel[c]), sigma_color);
                        double weight = spatial_weights[i * d + j] * range_weight;

                        // Accumuler la somme pondérée
                        filtered_value[c] += neighbor_pixel[c] * weight;
                        weight_sum[c] += weight;
                    }
                }
            }

            // Normaliser et stocker le résultat
            unsigned char *output_pixel = dst + (y * width + x) * channels;
            for (int c = 0; c < channels; c++) {
                output_pixel[c] = (unsigned char)(filtered_value[c] / (weight_sum[c] + 1e-6)); // Éviter la division par zéro
            }
        }
    }

    free(spatial_weights);
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

    // Ensure that image is not too small for bilateral filter (at least radius of d/2 around edges)
    if (width <= 5 || height <= 5) {
        printf("Image is too small for bilateral filter (at least 5x5 size needed).\n");
        stbi_image_free(image);
        return 1;
    }

    // Allocate memory for output image
    unsigned char *filtered_image = (unsigned char *)malloc(width * height * channels);
    if (!filtered_image) {
        printf("Memory allocation for filtered image failed!\n");
        stbi_image_free(image);
        return 1;
    }

    // Measure the time taken by the bilateral filter
    clock_t start_time = clock();
    
    // Apply the bilateral filter
    bilateral_filter(image, filtered_image, width, height, channels, 5, 75.0, 75.0);
    
    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Bilateral filter took %.3f seconds\n", elapsed_time);

    // Save the output image
    if (!stbi_write_png(argv[2], width, height, channels, filtered_image, width * channels)) {
        printf("Error saving the image!\n");
        free(filtered_image);
        stbi_image_free(image);
        return 1;
    }

    // Free memory
    stbi_image_free(image);
    free(filtered_image);

    printf("Bilateral filtering complete. Output saved as %s\n", argv[2]);
    return 0;
}