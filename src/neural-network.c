#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

#include "neural-network.h"

double sigmoid(double z) {
  return 1/(1 + exp(-z));
}

Layer *init_layer(unsigned int num_nodes) {
  int i;

  Layer *layer = malloc(sizeof(Layer));
  layer->num_nodes = num_nodes;
  layer->activation = &sigmoid;

  return layer;
}

void initialize_network(Neural_Network *network, size_t input_size, size_t output_size, size_t *hidden_layer_sizes, size_t num_hidden_layers) {
  int i;
  
  srand(time(NULL));

  network->num_layers = num_hidden_layers + 2;
  network->layers = malloc(sizeof(*network->layers) * network->num_layers);
  network->weights = malloc(sizeof(*network->weights) * (network->num_layers - 1));

  /* Initialize layers */
  network->layers[0] = init_layer(input_size);

  for (i = 0; i < num_hidden_layers; i++)
    network->layers[i + 1] = init_layer(hidden_layer_sizes[i]);

  network->layers[network->num_layers - 1] = init_layer(output_size);

  /* Initialize weights */
  for (i = 0; i < network->num_layers - 1; i++) {
    int r, c;
    size_t num_in = network->layers[i]->num_nodes;
    size_t num_out = network->layers[i + 1]->num_nodes;

    network->weights[i] = gsl_matrix_alloc(num_out, num_in + 1);

    for (r = 0; r < num_out; r++) {
      for (c = 0; c < num_in + 1; c++) {
        gsl_matrix_set(network->weights[i], r, c, rand() / (5.0 * RAND_MAX) - 0.1);
      }
    }
  }
}

void feed_forward(Neural_Network *network, double *input_values) {
  int i, l;
  Layer *cur_layer = network->layers[0];
  gsl_matrix *cur_vector = gsl_matrix_alloc(cur_layer->num_nodes + 1, 1);

  /* Initialize input vector including a bias value */
  gsl_matrix_set(cur_vector, 0, 0, 1);
  for (i = 1; i < cur_layer->num_nodes + 1; i++) 
    gsl_matrix_set(cur_vector, i, 0, input_values[i - 1]);

  for (l = 1; l < network->num_layers; l++) {
    gsl_matrix *weight_matrix = network->weights[l - 1];
    gsl_matrix *next_vector;
    
    cur_layer = network->layers[l];
    next_vector = gsl_matrix_alloc(cur_layer->num_nodes, 1);

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, weight_matrix, cur_vector, 0, next_vector);
    gsl_matrix_free(cur_vector);

    /* Apply activation function */
    for (i = 0; i < next_vector->size1; i++) {
      double activated_value = cur_layer->activation(gsl_matrix_get(next_vector, i, 0));
      gsl_matrix_set(next_vector, i, 0, activated_value);
    }

    /* Include a bias value if the current layer is not an output layer */
    if (l != network->num_layers - 1) {
      gsl_matrix *next_vector_with_bias = gsl_matrix_alloc(cur_layer->num_nodes + 1, 1);
      gsl_matrix_set(next_vector_with_bias, 0, 0, 1);
      for (i = 1; i < cur_layer->num_nodes + 1; i++) 
        gsl_matrix_set(next_vector_with_bias, i, 0, gsl_matrix_get(next_vector, i - 1, 0));
 
      gsl_matrix_free(next_vector);
      cur_vector = next_vector_with_bias;
    } else {
      cur_vector = next_vector;
    }
  }

  printf("Output is %dx%d\n", cur_vector->size1, cur_vector->size2);

  for (i = 0; i < cur_vector->size1; i++) {
    printf("%.02f\n", gsl_matrix_get(cur_vector, i, 0));
  }

  gsl_matrix_free(cur_vector);
}