#include <stdio.h>
#include "llm/core/matrix.h"

int main()
{
    Matrix m1 = new_matrix(2, 2);
    Matrix m2 = new_matrix(2, 2);
    randomize_matrix(m1);
    randomize_matrix(m2);
    Matrix m3 = dot_matrices(m1, m2);
    for (int i = 0; i < m3.rows * m3.cols; i++)
        printf("%f\n", m3.data[i]);
    return 0;
}