#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <gsl/gsl_blas.h>
#include "mnist-digits.h"

typedef struct {
  size_t num_nodes;
  double (*activation)(double);
  double (*activation_gradient)(double);
} Layer;

typedef struct {
  size_t num_layers;
  Layer **layers;
  gsl_matrix **weights;
} Neural_Network;

void initialize_network(Neural_Network *network, size_t input_size, size_t output_size, size_t *hidden_layer_sizes, size_t num_hidden_layers);

void train(Neural_Network *network, Image_Array *images, size_t batch_size);

unsigned char predict(Neural_Network *network, Image *image);

void load_weights(Neural_Network *network, char *filename);

void save_weights(Neural_Network *network, char *filename);

#endif