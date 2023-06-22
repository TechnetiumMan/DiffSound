

$$
\mathbf{M} \ddot{\mathbf{u}} + \mathbf{C} \dot{\mathbf{u}} + \mathbf{K} \mathbf{u} = \mathbf{f} \\
\ddot{\mathbf{u}} + \mathbf{M}^{-1} \mathbf{C} \dot{\mathbf{u}} + \mathbf{M}^{-1} \mathbf{K} \mathbf{u} = \mathbf{M}^{-1} \mathbf{f} \\
$$
can be rewrite as
$$
\begin{bmatrix} \ddot{\mathbf{u}} \\ \dot{\mathbf{u}} \end{bmatrix} = \begin{bmatrix} -\mathbf{M}^{-1} \mathbf{C} & -\mathbf{M}^{-1} \mathbf{K} \\ \mathbf{I} & \mathbf{0} \end{bmatrix} \begin{bmatrix} \dot{\mathbf{u}} \\ \mathbf{u} \end{bmatrix} + \begin{bmatrix} \mathbf{M}^{-1} \mathbf{f} \\ \mathbf{0} \end{bmatrix}
$$