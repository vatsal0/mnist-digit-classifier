#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_blas.h>

#include "neural-network.h"

double sigmoid(double z) {
  return 1/(1 + exp(-z));
}

double sigmoid_gradient(double z) {
  return sigmoid(z) * (1 - sigmoid(z));
}

Layer *init_layer(unsigned int num_nodes) {
  int i;

  Layer *layer = malloc(sizeof(Layer));
  layer->num_nodes = num_nodes;
  layer->node_values = malloc(sizeof(*layer->node_values) * num_nodes);
  layer->activation = &sigmoid;
  layer->activation_gradient = &sigmoid_gradient;

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

double backpropagate(Neural_Network *network, double *input_values, unsigned int expected_output) {
  int i, l;
  double cost = 0;
  double lambda = 0.1 / (network->num_layers - 1);
  Layer *cur_layer = network->layers[0];
  gsl_matrix *cur_vector = gsl_matrix_alloc(cur_layer->num_nodes + 1, 1);
  gsl_matrix *expected_vector;
  gsl_matrix **deltas = malloc(sizeof(gsl_matrix *) * network->num_layers);

  /* Initialize input vector including a bias value */
  gsl_matrix_set(cur_vector, 0, 0, 1);
  for (i = 0; i < cur_layer->num_nodes; i++) 
    gsl_matrix_set(cur_vector, i + 1, 0, input_values[i]);
    cur_layer->node_values[i] = input_values[i];

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
      cur_layer->node_values[i] = gsl_matrix_get(next_vector, i, 0);
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

  deltas[network->num_layers - 1] = gsl_matrix_alloc(cur_vector->size1, 1);

  for (i = 0; i < cur_vector->size1; i++) {
    double h = gsl_matrix_get(cur_vector, i, 0);
    double y = i == expected_output;

    cost += -y * log(h) - (1 - y) * log(1 - h);
    gsl_matrix_set(deltas[network->num_layers - 1], i, 0, h - y);
  }

  for (l = 1; l < network->num_layers; l++) {
    int r, c;
    gsl_matrix *weight_matrix = network->weights[l - 1];
    for (r = 0; r < weight_matrix->size1; r++) {
      for (c = 0; c < weight_matrix->size2; c++) {
        cost += lambda * gsl_matrix_get(weight_matrix, r, c) * gsl_matrix_get(weight_matrix, r, c);
      }
    }
  }

  for (l = network->num_layers - 1; l > 1; l--) {
    Layer *back_layer = network->layers[l - 1];
    gsl_matrix *weight_matrix = network->weights[l - 1];
    gsl_matrix *activated_delta = gsl_matrix_alloc(back_layer->num_nodes, 1);
    deltas[l - 1] = gsl_matrix_alloc(back_layer->num_nodes + 1, 1);

    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, weight_matrix, deltas[l], 0, deltas[l - 1]);
    for (i = 0; i < back_layer->num_nodes; i++) {
      gsl_matrix_set(activated_delta, i, 0, gsl_matrix_get(deltas[l - 1], i + 1, 0) * back_layer->activation_gradient(back_layer->node_values[i]));
    }
    gsl_matrix_free(deltas[l - 1]);
    deltas[l - 1] = activated_delta;
  }

  for (l = 0; l < network->num_layers - 1; l++) {
    gsl_matrix *weight_matrix = network->weights[l];
    gsl_matrix *grad_matrix = gsl_matrix_alloc(weight_matrix->size1, weight_matrix->size2);
    cur_layer = network->layers[l];
    int r, c;

    for (r = 0; r < weight_matrix->size1; r++) {
      gsl_matrix_set(grad_matrix, r, 0, 0);
      for (c = 1; c < weight_matrix->size2; c++) {
        gsl_matrix_set(grad_matrix, r, c, gsl_matrix_get(deltas[l + 1], r, 0) * cur_layer->activation(cur_layer->node_values[c]));
      }
    }

    gsl_matrix_sub(weight_matrix, grad_matrix);
    gsl_matrix_free(grad_matrix);
  }

  for (l = 1; l < network->num_layers; l++)
    gsl_matrix_free(deltas[l]);
  free(deltas);

  gsl_matrix_free(cur_vector);

  return cost;
}