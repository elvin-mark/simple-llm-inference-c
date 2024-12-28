#include <stdlib.h>
#include <stdio.h>
#include "llm/core/matrix.h"
#include "llm/core/utils.h"

Matrix new_matrix(int rows, int cols)
{
    Matrix res = {
        data : malloc(sizeof(float) * rows * cols),
        rows : rows,
        cols : cols,
        T : 0
    };
    return res;
}

float get_element(Matrix t, int i, int j)
{
    if (t.rows == 1)
        i = 0;
    if (t.cols == 1)
        j = 0;
    if (t.T)
        return t.data[j * t.rows + j];
    return t.data[i * t.cols + j];
}

void set_element(Matrix t, int i, int j, float elem)
{
    if (t.T)
        t.data[j * t.rows + j] = elem;
    t.data[i * t.cols + j] = elem;
}

void randomize_matrix(Matrix m)
{
    for (int i = 0; i < m.rows * m.cols; i++)
        m.data[i] = (1.0 * rand()) / RAND_MAX;
}

Matrix add_matrices(Matrix m1, Matrix m2)
{
    assert(m1.rows == m2.rows || m1.rows == 1 || m2.rows == 1, "matrices' rows do not match");
    assert(m1.cols == m2.cols || m1.cols == 1 || m2.cols == 1, "matrices' cols do not match");
    int rows = m1.rows > m2.rows ? m1.rows : m2.rows;
    int cols = m1.cols > m2.cols ? m1.cols : m2.cols;
    Matrix res = new_matrix(rows, cols);
    float s;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            s = get_element(m1, i, j) + get_element(m2, i, j);
            set_element(res, i, j, s);
        }
    }
    return res;
}

Matrix dot_matrices(Matrix m1, Matrix m2)
{
    assert(m1.cols == m2.rows, "not appropiate shape");
    Matrix res = new_matrix(m1.rows, m2.cols);
    float s;
    for (int i = 0; i < m1.rows; i++)
    {
        for (int j = 0; j < m2.cols; j++)
        {
            s = 0;
            for (int k = 0; k < m1.cols; k++)
                s += get_element(m1, i, k) * get_element(m2, k, j);
            set_element(res, i, j, s);
        }
    }
    return res;
}

Matrix slice(Matrix m, int *indices, int num_indices)
{
    Matrix res = new_matrix(num_indices, m.cols);
    float s;
    for (int i = 0; i < num_indices; i++)
    {
        for (int j = 0; j < m.cols; j++)
        {
            s = get_element(m, indices[i], j);
            set_element(res, i, j, s);
        }
    }
    return res;
}

Matrix apply_fn(Matrix m, float (*fn)(float))
{
    Matrix res = new_matrix(m.rows, m.cols);
    for (int i = 0; i < m.rows * m.cols; i++)
        res.data[i] = fn(m.data[i]);
    return res;
}