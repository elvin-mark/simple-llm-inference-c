#ifndef NN_H
#define NN_H

#include "llm/core/matrix.h"

Matrix linear(Matrix W, Matrix b, Matrix in);

Matrix layer_norm(Matrix m, Matrix g, Matrix b);

Matrix softmax(Matrix m);

Matrix act_fn_sigmoid(Matrix m);

Matrix act_fn_tanh(Matrix m);

Matrix act_fn_gelu(Matrix m);

#endif