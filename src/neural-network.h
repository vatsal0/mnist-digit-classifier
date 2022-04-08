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

/* Initializes the fields for the passed neural network with the given amount of input nodes, output nodes, and hidden layers. 
 * The weights for each node are randomly assigned to a value between -0.1 and 0.1. For now, layer activation functions default to the sigmoid function. */
void initialize_network(Neural_Network *network, size_t input_size, size_t output_size, size_t *hidden_layer_sizes, size_t num_hidden_layers);

/* Trains the neural network, splitting the training set into the specified number of batches.
 * The cost of each batch is outputted to stdout before the backpropagation algorithm runs. */ 
void train(Neural_Network *network, Image_Array *images, size_t num_batches);

/* Returns the digit with the highest probability of prediction by the neural network for the given image. 
 * The calculations for this function differ from training the network because there is only one example to be fed forward. */
unsigned char predict(Neural_Network *network, Image *image);

/* Loads weights into the neural network from a file. The file is assumed to be a contiguous array of doubles in row major order.
 * The total length of the array stored in the file must align with the total number of elements across all of the network's weight matrices. */
void load_weights(Neural_Network *network, char *filename);

/* Saves the weights of the neural network into a binary file. The file will store an array of n doubles, 
 * where n is the total number of elements in all of the network's weight matrices. */
void save_weights(Neural_Network *network, char *filename);

/* Frees all memory associated with the network, including the pointer to the network itself. */
void free_network(Neural_Network *network);

#endif