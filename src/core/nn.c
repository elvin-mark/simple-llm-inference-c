#include "llm/core/nn.h"
#include "llm/core/matrix.h"
#include <math.h>

float _sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}

float _tanh(float x)
{
    return (1 - exp(-x)) / (1 + exp(-x));
}

float _gelu(float x)
{
    return x * _sigmoid(1.702 * x);
}

Matrix linear(Matrix W, Matrix b, Matrix in)
{
    return add_matrices(dot_matrices(in, W), b);
}

Matrix layer_norm(Matrix m, Matrix g, Matrix b)
{
}

Matrix softmax(Matrix m)
{
}

Matrix act_fn_sigmoid(Matrix m)
{
    return apply_fn(m, _sigmoid);
}

Matrix act_fn_tanh(Matrix m)
{
    return apply_fn(m, _tanh);
}

Matrix act_fn_gelu(Matrix m)
{
    return apply_fn(m, _gelu);
}