#ifndef __CRITERION_H__
#define __CRITERION_H__
#include "matrix.h"

typedef struct criterion {
  matrix *(*criterion_output)(matrix *, matrix *);
  matrix *(*criterion_gradient)(matrix *, matrix *);
  matrix *gradient;
  matrix *output;
} criterion;

criterion *criterion_create(
  matrix *(*criterion_output)(matrix *, matrix *),
  matrix *(*criterion_gradient)(matrix *, matrix *)
);
matrix *criterion_forward(criterion *c, matrix *output, matrix *targets);
matrix *criterion_backward(criterion *c, matrix *output, matrix *targets);
matrix *criterion_forward_mse(matrix *output, matrix *target);
matrix *criterion_backward_mse(matrix *output, matrix *target);
void criterion_free(criterion *c);

#endif
