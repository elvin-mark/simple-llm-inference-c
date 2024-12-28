#ifndef MATRIX_H
#define MATRIX_H

typedef struct Matrix
{
    float *data;
    int rows;
    int cols;
    int T;
} Matrix;

Matrix new_matrix(int rows, int cols);
void randomize_matrix(Matrix m);
Matrix add_matrices(Matrix m1, Matrix m2);
Matrix dot_matrices(Matrix m1, Matrix m2);
Matrix slice(Matrix m, int *indices, int num_indices);
Matrix apply_fn(Matrix m, float (*fn)(float));

#endif