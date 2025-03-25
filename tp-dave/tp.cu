#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>


// Macro pour vérifier les erreurs CUDA
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

// Filtre bilatéral manuel
void bilateral_filter(unsigned char *src, unsigned char *dst, int width, int height, int channels, int d, double sigma_color, double sigma_space) {
    int radius = d / 2;

    // Précalcul des poids spatiaux Gaussiens
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

    // Traitement de l'image
    for (int y = radius; y < height - radius; y++) {
        for (int x = radius; x < width - radius; x++) {
            double weight_sum[3] = {0.0, 0.0, 0.0};
            double filtered_value[3] = {0.0, 0.0, 0.0};

            // Pointeur vers le pixel central
            unsigned char *center_pixel = src + (y * width + x) * channels;

            // Parcours de la fenêtre locale
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    int nx = x + j - radius;
                    int ny = y + i - radius;

                    // Vérification des limites pour rester dans l'image
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                        continue;
                    }

                    // Pointeur vers le pixel voisin
                    unsigned char *neighbor_pixel = src + (ny * width + nx) * channels;

                    for (int c = 0; c < channels; c++) {
                        // Calcul du poids de portée
                        double range_weight = gaussian(abs(neighbor_pixel[c] - center_pixel[c]), sigma_color);
                        double weight = spatial_weights[i * d + j] * range_weight;

                        // Accumulation de la somme pondérée
                        filtered_value[c] += neighbor_pixel[c] * weight;
                        weight_sum[c] += weight;
                    }
                }
            }

            // Normalisation et stockage du résultat
            unsigned char *output_pixel = dst + (y * width + x) * channels;
            for (int c = 0; c < channels; c++) {
                output_pixel[c] = (unsigned char)(filtered_value[c] / (weight_sum[c] + 1e-6)); // Éviter la division par zéro
            }
        }
    }

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

    // Vérification que l'image n'est pas trop petite pour le filtre bilatéral
    if (width <= 5 || height <= 5) {
        printf("L'image est trop petite pour le filtre bilatéral (taille minimale 5x5 requise).\n");
        stbi_image_free(image);
        return 1;
    }

    // Allocation mémoire pour l'image filtrée
    unsigned char *filtered_image = (unsigned char *)malloc(width * height * channels);
    if (!filtered_image) {
        printf("Échec de l'allocation mémoire pour l'image filtrée !\n");
        stbi_image_free(image);
        return 1;
    }
    
    // Application du filtre bilatéral
    bilateral_filter(image, filtered_image, width, height, channels, 5, 75.0, 75.0);

    // Sauvegarde de l'image de sortie
    if (!stbi_write_png(argv[2], width, height, channels, filtered_image, width * channels)) {
        printf("Erreur lors de la sauvegarde de l'image !\n");
        free(filtered_image);
        stbi_image_free(image);
        return 1;
    }

    // Libération de la mémoire
    stbi_image_free(image);
    free(filtered_image);

    printf("Filtrage bilatéral terminé. Sortie sauvegardée sous %s\n", argv[2]);
    return 0;
}