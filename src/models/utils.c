#include "llm/core/matrix.h"
#include "llm/models/utils.h"

LinearLayer create_linear_layer(int in, int out)
{
    LinearLayer l = {
        W : new_matrix(in, out),
        b : new_matrix(1, out)
    };
    return l;
}

LayerNorm create_layer_norm_layer(int dim)
{
    LayerNorm l = {
        g : new_matrix(1, dim),
        b : new_matrix(1, dim)
    };
    return l;
}