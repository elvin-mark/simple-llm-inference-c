#ifndef MODEL_UTILS_H
#define MODEL_UTILS_H

#include "llm/core/matrix.h"

typedef struct LinearLayer
{
    Matrix W;
    Matrix b;
} LinearLayer;

typedef struct LayerNorm
{
    Matrix g;
    Matrix b
} LayerNorm;

LinearLayer create_linear_layer(int in, int out);
LayerNorm create_layer_norm_layer(int dim);
#endif