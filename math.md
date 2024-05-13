matrix multiplication
- $a_{ij}$
    - entry from a matrix where
        - i: row of the matrix
        - j: col of the matrix

- $AB=C$
    - prerequisite: number of rows in a = number of cols in b
    - the size of the c = number of rows in A * number of cols in B
    $$
        \begin{bmatrix}
            a_{00}, a_{01}\\
            a_{10}, a_{11}
        \end{bmatrix}
        \begin{bmatrix}
            b_{00}, b_{01}\\
            b_{10}, b_{11}
        \end{bmatrix}
        =
        \begin{bmatrix}
            c_{00}, c_{01}\\
            c_{10}, c_{11}
        \end{bmatrix}
    $$
    - each entry, ij, in c is dot product of i row in a and j col in b. That is:
        $$
            c_{ij} = a[i,:] \cdot b[:, j]
        $$
- basis
    - a set of vectors that forms any element in a space with linear combination of them
        - e.g. In space, any point can be expressed by vector of x, y, z
            - xyz unit vectors are the basis
            - linear combination among unit vector is coordiate

- column vector(coordinate vector) tell us how current vector is formed by the basis in the space
    - e.g. we have a space x with n dimension in it
        $$
            x =\begin{bmatrix}
                a_{1},\\
                a_{12} \\
                ..\\
                a_{n}
            \end{bmatrix}
        $$

- We need linear map to transform current column vector in x space to y space.  A has the size of m x n:
$$
        A =\begin{bmatrix}
            a_{11}, ..., a_{n1}\\
            a_{21}, ..., a_{n2} \\
            ..\\
            a_{m1}, ..., a_{mn}
        \end{bmatrix}
$$

- To express this transformation, we have
$$
    y = Ax
$$

- To have one more transformation with B having size of p x m
$$
    z = BAx = (BA)x = B(Ax)
$$
- we can rotate basis(coordinate) on a plane with the map
$$
        A =\begin{bmatrix}
            \cos\alpha, -\sin\alpha \\
            \sin\alpha, -\cos\alpha \\
        \end{bmatrix}
$$


ref
1. [wiki](https://en.wikipedia.org/wiki/Matrix_multiplication)
