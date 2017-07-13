#include "matrix.h"
#include "criterion.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * criterion_create
 */
criterion *criterion_create(
  matrix *(*criterion_output)(matrix *, matrix *),
  matrix *(*criterion_gradient)(matrix *, matrix *)
) {
  criterion *c;
  if ((c = malloc(sizeof(*c))) == NULL) {
    perror("Out of memory\n");
    return NULL;
  }
  c->criterion_output = criterion_output;
  c->criterion_gradient = criterion_gradient;
  c->output = NULL;
  c->gradient = NULL;
  return c;
}

/*
 * criterion_forward calculates the 1d-loss
 */
matrix *criterion_forward(criterion *c, matrix *output, matrix *target) {
  if (c->output == NULL) {
    c->output = malloc(sizeof(*output));
  }
  *c->output = *c->criterion_output(output, target);
  return matrix_copy(c->output);
}

/*
 * criterion_backward calculates the gradient w.r.t the output
 */
matrix *criterion_backward(criterion *c, matrix *output, matrix *target) {
  if (c->gradient == NULL) {
    c->gradient = malloc(sizeof(*output));
  }
  *c->gradient = *c->criterion_gradient(output, target);
  return matrix_copy(c->gradient);
}

/*
 * criterion_forward_mse, calculate the mse loss
 */
matrix *criterion_forward_mse(matrix *output, matrix *target) {
  int i, j;
  float error = 0.0, sum = 0.0;
  matrix *c = matrix_create(1, 1, NULL);
  for (i = 0; i < output->rows; i++) {
    for (j = 0; j < output->columns; j++) {
      error = output->data[i][j] - target->data[i][j];
      sum += error * error;
    }
  }
  c->data[0][0] = sum / (float)output->rows;
  return c;
}

/*
 * criterion_backward_mse
 */
matrix *criterion_backward_mse(matrix *output, matrix *target) {
  int i, j;
  float norm = 2.0 / (float)output->rows;
  matrix *gradient = matrix_create(output->rows, output->columns, NULL);
  for (i = 0; i < output->rows; i++) {
    for (j = 0; j < output->columns; j++) {
      gradient->data[i][j] = norm * (output->data[i][j] - target->data[i][j]);
    }
  }
  return gradient;
}

/*
 * criterion_free
 */
void criterion_free(criterion *c) {
  free(c->gradient);
  free(c->output);
  free(c);
  c = NULL;
}
