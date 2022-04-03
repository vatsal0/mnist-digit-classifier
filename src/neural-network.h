#include <gsl/gsl_blas.h>

typedef struct {
  size_t num_nodes;
  double *node_values;
  double (*activation)(double);
  double (*activation_gradient)(double);
} Layer;

typedef struct {
  size_t num_layers;
  Layer **layers;
  gsl_matrix **weights;
} Neural_Network;

void initialize_network(Neural_Network *network, size_t input_size, size_t output_size, size_t *hidden_layer_sizes, size_t num_hidden_layers);

double backpropagate(Neural_Network *network, double *input_values, unsigned int expected_output);