#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_blas.h>

#include "neural-network.h"

#ifndef LAMBDA
#define LAMBDA 1
#endif

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

void train(Neural_Network *network, Image_Array *array, size_t batch_size) {
  int i_batch, i_example, i_layer, i_value;
  Layer *cur_layer, *next_layer;
  gsl_matrix_view cur_layer_matrix, next_layer_matrix;
  gsl_matrix **gradients = calloc(network->num_layers - 1, sizeof(gsl_matrix *));
  double **deltas = calloc(network->num_layers, sizeof(double *));
  double **layer_values = calloc(network->num_layers, sizeof(double *));
  double **layer_activations = calloc(network->num_layers, sizeof(double *));

  /* Initialize layer value matrices */ 
  for (i_layer = 0; i_layer < network->num_layers; i_layer++) {
    cur_layer = network->layers[i_layer];

    layer_values[i_layer] = calloc(cur_layer->num_nodes * batch_size, sizeof(**layer_values));
    layer_activations[i_layer] = calloc(cur_layer->num_nodes * batch_size, sizeof(**layer_values));
  }

  for (i_batch = 0; i_batch < array->num_images; i_batch += batch_size) {
    double cost;
    size_t input_size = network->layers[0]->num_nodes;
    size_t output_size = network->layers[network->num_layers - 1]->num_nodes;

    /* Load input values into first layer */
    for (i_example = 0; i_example < batch_size; i_example++) {
      for (i_value = 0; i_value < input_size; i_value++)
        layer_values[0][i_example * input_size + i_value] = (double) array->images[i_batch + i_example]->pixels[i_value];
    }

    /* The activation for the input layer is just going to be normalizing the batch i.e. dividing by 255 */
    for (i_value = 0; i_value < input_size * batch_size; i_value++)
      layer_activations[0][i_value] = layer_values[0][i_value] / 255.0;


    /* Calculate output through forward propagation */
    for (i_layer = 0; i_layer < network->num_layers - 1; i_layer++) {
      gsl_matrix *layer_with_bias;

      cur_layer = network->layers[i_layer];
      next_layer = network->layers[i_layer + 1];

      /* Matrix is converted in row major order, and each training example is contiguous in memory so a row represents an example */
      /* These matrices have to be transposed for multiplication, because examples must be column vectors to match the weight matrices' dimensions */
      cur_layer_matrix = gsl_matrix_view_array(layer_activations[i_layer], batch_size, cur_layer->num_nodes);
      next_layer_matrix = gsl_matrix_view_array(layer_values[i_layer + 1], batch_size, next_layer->num_nodes);
      layer_with_bias = gsl_matrix_alloc(cur_layer->num_nodes + 1, batch_size);

      for(i_example = 0; i_example < batch_size; i_example++) {
        /* Add a bias value to the current example. */
        gsl_matrix_set(layer_with_bias, 0, i_example, 1);

        /* The original matrix is pushed down one row to make space for the bias term. Also, the indices are switched when copying to transpose the matrix. */
        for (i_value = 0; i_value < cur_layer->num_nodes; i_value++)
          gsl_matrix_set(layer_with_bias, i_value + 1, i_example, gsl_matrix_get(&cur_layer_matrix.matrix, i_example, i_value));
      }

      /* Calculate the values for the next layer using the weight matrix. */
      gsl_blas_dgemm(CblasTrans, CblasTrans, 1, layer_with_bias, network->weights[i_layer], 0, &next_layer_matrix.matrix);

      /* Apply the activation function on every entry of the next layer to set it up for the next iteration. */
      for (i_value = 0; i_value < next_layer->num_nodes * batch_size; i_value++)
          layer_activations[i_layer + 1][i_value] = next_layer->activation(layer_values[i_layer + 1][i_value]);

      gsl_matrix_free(layer_with_bias);
    }

    /* Calculate cost function for the output */
    cost = 0;

    for (i_example = 0; i_example < batch_size; i_example++) {
      unsigned char label = array->images[i_batch + i_example]->label;

      for (i_value = 0; i_value < output_size; i_value++) {
        double prediction = layer_activations[network->num_layers - 1][i_example * output_size + i_value];
        if (i_value == label)
          cost += -log(prediction);
        else
          cost += -log(1 - prediction);
      }
    }

    /* Add regularization to the cost function by adding square sum of weight values */
    for (i_layer = 0; i_layer < network->num_layers - 1; i_layer++){
      int r,c;
      gsl_matrix *cur_weights = network->weights[i_layer];

      for (r = 0; r < cur_weights->size1; r++) {
        /* Bias weight (column 0) is ignored in regularization */
        for (c = 1; c < cur_weights->size2; c++) {
          double weight = gsl_matrix_get(cur_weights, r, c);
          cost += weight * weight * LAMBDA / 2.0;
        }
      }
    }

    cost /= batch_size;
    printf("Cost for batch %lu: %.02f\n", i_batch / batch_size + 1, cost);

    /* Initialize gradient matrices and delta vectors */
    for (i_layer = 0; i_layer < network->num_layers - 1; i_layer++) {
      gradients[i_layer] = gsl_matrix_alloc(network->weights[i_layer]->size1, network->weights[i_layer]->size2);
      deltas[i_layer + 1] = calloc(network->weights[i_layer]->size1, sizeof(**deltas));
    }


    /* Adjust weights through backpropagation */
    for (i_example = 0; i_example < batch_size; i_example++) {
      /* Calculate output delta */
      unsigned char label = array->images[i_batch + i_example]->label;
      for (i_value = 0; i_value < output_size; i_value++) {
        double prediction = layer_activations[network->num_layers - 1][i_example * output_size + i_value];
        deltas[network->num_layers - 1][i_value] = prediction - (i_value == label);
      }

      /* Calculate deltas for previous layers */
      for (i_layer = network->num_layers - 2; i_layer > 0; i_layer--) {
        int num_nodes = network->layers[i_layer]->num_nodes;
        gsl_matrix_view next_delta_matrix = gsl_matrix_view_array(deltas[i_layer + 1], network->layers[i_layer + 1]->num_nodes, 1);
        gsl_matrix *cur_delta_matrix = gsl_matrix_alloc(num_nodes + 1, 1);

        gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, network->weights[i_layer], &next_delta_matrix.matrix, 0, cur_delta_matrix);

        for (i_value = 0; i_value < num_nodes; i_value++)
          deltas[i_layer][i_value] = gsl_matrix_get(cur_delta_matrix, i_value + 1, 0) * sigmoid_gradient(layer_values[i_layer][i_example * num_nodes + i_value]);

        gsl_matrix_free(cur_delta_matrix);
      }

      for (i_layer = 0; i_layer < network->num_layers - 1; i_layer++) {
        int num_nodes = network->layers[i_layer]->num_nodes;
        int r,c;
        gsl_matrix *cur_weights = network->weights[i_layer];
        gsl_matrix *cur_activations = gsl_matrix_alloc(num_nodes + 1, 1);
        gsl_matrix_view next_delta_matrix = gsl_matrix_view_array(deltas[i_layer + 1], network->layers[i_layer + 1]->num_nodes, 1);

        gsl_matrix_set(cur_activations, 0, 0, 1);
        for (i_value = 0; i_value < num_nodes; i_value++)
          gsl_matrix_set(cur_activations, i_value + 1, 0, layer_activations[i_layer][i_example * num_nodes + i_value]);

        /* Calculate the gradient for the current example with the delta and transpose of actiations */
        gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1, &next_delta_matrix.matrix, cur_activations, 0, gradients[i_layer]);

        /* Regularize the gradient by adding current weight values (ignoring bias again) */
        for (r = 0; r < cur_weights->size1; r++) {
          for (c = 1; c < cur_weights->size2; c++)
            gsl_matrix_set(gradients[i_layer], r, c, gsl_matrix_get(gradients[i_layer], r, c) + gsl_matrix_get(cur_weights, r, c) * LAMBDA);
        }

        /* Scale the matrix for the single example */
        gsl_matrix_scale(gradients[i_layer], 1.0/batch_size);
        gsl_matrix_sub(cur_weights, gradients[i_layer]);

        gsl_matrix_free(cur_activations);
      }
    }
  }
}