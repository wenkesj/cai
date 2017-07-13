#ifndef __MATRIX_H_
#define __MATRIX_H_

typedef struct matrix {
  int rows;
  int columns;
  float **data;
} matrix;

float matrix_zeros(int i, int j);
float matrix_ones(int i, int j);
float matrix_random(int i, int j);
matrix *matrix_create(int rows, int columns, float (*initialize_function)(int, int));
matrix *matrix_multiply(matrix *a, matrix *b);
matrix *matrix_add(matrix *a, matrix *b);
matrix *matrix_scale(matrix *a, float b);
matrix *matrix_transpose(matrix *m);
matrix *matrix_copy(matrix *m);
void matrix_print(char *name, matrix *input);
void matrix_free(matrix *m);

#endif
