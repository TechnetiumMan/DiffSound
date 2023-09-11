https://zhuanlan.zhihu.com/p/56159173
https://cs.stanford.edu/~ambrad/adjoint_tutorial.pdf

很多时候，我们会希望对一个系统的参数进行优化，此时系统所遵循的物理规律（一般用PDE形式表示）我们是知道的。这类问题称为PDE-constrained optimization，有许多应用场景。解决这类问题的一类方法称为adjoint method.

例如，假设系统参数为p，变量为x，满足(可能是微分)方程g(x,p)=0,我们想要优化p使得最小化loss f(x,p)。
一个自然的想法是计算$\frac{df}{dp}$，这样就可以用梯度优化算法了！但是，对于微分方程，怎么算？
暴力方法是，对参数的每个维度取小变化差分p'=p+Δp，硬算梯度！但是这样的话，一次梯度优化要进行Ndim次模拟，不可接受！或者让计算全过程可微(就是我们之前的方法)

在adjoint method中，我们使用拉格朗日乘子法求f的极小值。定义
$L=f(x,p)+\lambda g(x,p)$
则有：
$\frac{\partial L}{\partial p}=(\frac{\partial f}{\partial x}+\lambda \frac{\partial g}{\partial x})\frac{\partial x}{\partial p}+\frac{\partial f}{\partial p}+\lambda \frac{\partial g}{\partial p}$
我们取乘子$\lambda$使得$\frac{\partial L}{\partial x}=(\frac{\partial f}{\partial x}+\lambda \frac{\partial g}{\partial x})=0$，则可得
$\frac{df}{dp}=\frac{\partial L}{\partial p}=\frac{\partial f}{\partial p}+\lambda \frac{\partial g}{\partial p}$
（这是因为全导数$\frac{dg}{dp}=0$）

### 视频教程笔记
https://www.youtube.com/watch?v=k6s2G5MZv-I
我们设$u(t)$为系统中微分方程$\frac{du}{dt}=f(u,t,\theta)$的解（n维向量），$\theta$为系统参数，$J(u;\theta)=\int g(u;\theta)dt$为loss function，则loss关于系统参数的梯度可如下表示：
$\frac{dJ}{d\theta}=\int (\frac{\partial g}{\partial \theta}+\frac{\partial g}{\partial u}\frac{du}{d\theta})dt$
在以上方程中，$\frac{\partial g}{\partial \theta}$与$\frac{\partial g}{\partial u}$两项取决于loss function的定义，因此问题的关键在于对$\frac{du}{d\theta}$项的计算。
我们有两个思路：1.基于正向(forward)的方法；2.基于伴随(adjoint)/反向(backward)的方法。

首先考虑正向方法。在正向方法中，我们将$\theta$分成其p个分量：
$\frac{du}{d\theta}=[\frac{du}{d\theta_1}\dots\frac{du}{d\theta_p}]_{n×p}$
考虑系统微分方程：
$\frac{du}{dt}=f,u(0)=u_0$
两边同时分别对$\theta$的p个分量求导：
$\frac{d}{d\theta_i}\frac{du}{dt}=\frac{d}{d\theta_i}f, i=1, \dots, p$
该方程左边交换求导顺序，右边直接求导：
$\frac{d}{dt}\frac{du}{d\theta_i}=\frac{\partial f}{\partial u}\frac{\partial u}{\partial \theta_i}+\frac{\partial f}{\partial \theta_i}$
注意到，我们想求的变量是$\frac{du}{d\theta_i}(t)$，上式就是关于该变量的微分方程！将其解出即可达到目的。
然而，如果$\theta$的维度p较大，例如p=1000，我们就需要求解一千个微分方程(步进模拟一千次)！这开销巨大！

现在我们考虑伴随方法。我们的目的是，在满足方程$\frac{du}{dt}=f(u,t,\theta)$的条件下，最小化loss $J(u;\theta) = \int g(u;\theta)dt$.
因此我们使用拉格朗日乘子法，定义函数：
$L(u;\lambda;\theta) = J(u;\theta)+\int^T_0 (\lambda(t)(f-\frac{du}{dt}))dt$
求其关于$\theta$的导数可以得到：
$\frac{dL}{d\theta}=\int[\frac{\partial g}{\partial \theta}+\lambda(t)\frac{\partial f}{\partial \theta}+(\frac{\partial g}{\partial u}+\lambda(t)\frac{\partial f}{\partial u}-\lambda(t)\frac{d}{dt})\frac{du}{d\theta}]dt$
上式中，由正向方法可知$\frac{du}{d\theta}$是非常难算的！因此我们想要通过取乘子项$\lambda(t)$，让$\frac{du}{d\theta}$的系数为0，这样就不用算它了！
首先考虑最后一项,使用分部积分处理：
$\int^T_0 (-\lambda(t)\frac{d}{dt}\frac{du}{d\theta})dt$
$=[-\lambda(t)\frac{du}{d\theta}]^T_0+\int^T_0 \frac{d\lambda}{dt}\frac{du}{d\theta}dt$

将其带回原式，原式中$\frac{du}{d\theta}$的系数变为
$(\frac{\partial g}{\partial u}+\lambda(t)\frac{\partial f}{\partial u}+\frac{d\lambda}{dt})$
并多出了两个常数项
$\lambda(0)\frac{du}{d\theta}(0)-\lambda(T)\frac{du}{d\theta}(T)$
第一个常数项可以直接从初始状态得到，不考虑。

现在，我们的目标是，让$\frac{du}{d\theta}$系数为0，并且第二个常数项为0作为边界条件。基于此，我们构造变量为$\lambda$的微分方程：
$\frac{\partial g}{\partial u}+\lambda\frac{\partial f}{\partial u}+\frac{d\lambda}{dt}=0, \lambda(T)=0$
注意：这个方程相比于最初的关于u的微分方程，它是时间逆向的，从T往0算的！因此又称为backward。

截至目前，我们已经计算出了乘子项$\lambda$并算出了$\frac{dL}{d\theta}$。我们要算的目标梯度是$\frac{dJ}{d\theta}$，然而，由于
$L=J+\int^T_0 (\lambda(f-\frac{du}{dt}))dt$,
而由问题定义知$f=\frac{du}{dt}$，$\lambda$的系数为0，因此$\lambda$的选择可以是任意的，并且有$\frac{dJ}{d\theta}=\frac{dL}{d\theta}$。

### 我们的问题
线性情况：
$M\ddot{u}+C\dot{u}+Ku=0, u(0)=u_0$,
其中$M=M(\theta),C=C(\theta),K=K(\theta)$

RK4算法中，我们有：
$\begin{bmatrix} \ddot{\mathbf{u}} \\ \dot{\mathbf{u}} \end{bmatrix} = \begin{bmatrix} -\mathbf{M}^{-1} \mathbf{C} & -\mathbf{M}^{-1} \mathbf{K} \\ \mathbf{I} & \mathbf{0} \end{bmatrix} \begin{bmatrix} \dot{\mathbf{u}} \\ \mathbf{u} \end{bmatrix} + \begin{bmatrix} \mathbf{M}^{-1} \mathbf{f} \\ \mathbf{0} \end{bmatrix}$
令$x=\begin{bmatrix} \dot{\mathbf{u}} \\ {\mathbf{u}} \end{bmatrix},A=\begin{bmatrix} -\mathbf{M}^{-1} \mathbf{C} & -\mathbf{M}^{-1} \mathbf{K} \\ \mathbf{I} & \mathbf{0} \end{bmatrix},b=\begin{bmatrix} \mathbf{M}^{-1} \mathbf{f} \\ \mathbf{0} \end{bmatrix}$
则原方程化为：
$\dot{x}=Ax+b$

令$f=Ax+b$则有：
$L=J+\int^T_0(\lambda(t)^T(f-\dot{x}))dt$
$\frac{dL}{d\theta}=\int[\frac{\partial g}{\partial \theta}+\lambda(t)^T\frac{\partial f}{\partial \theta}+(\frac{\partial g}{\partial x}+\lambda(t)^T\frac{\partial f}{\partial x}-\lambda(t)^T\frac{d}{dt})\frac{dx}{d\theta}]dt$
$=\int[\lambda(t)^T\frac{\partial f}{\partial \theta}+(\frac{\partial g}{\partial x}+\lambda(t)^TA-\lambda(t)^T\frac{d}{dt})\frac{dx}{d\theta}]dt$
$(\frac{\partial g}{\partial \theta}=0,\frac{\partial f}{\partial x}=A)$

其中括号内最后一项
$\int^T_0 (-\lambda(t)^T\frac{d}{dt}\frac{dx}{d\theta})dt$
$=[-\lambda(t)^T\frac{dx}{d\theta}]^T_0+\int^T_0 \frac{d\lambda}{dt}\frac{dx}{d\theta}dt$
$=-\lambda(t)^T\frac{dx}{d\theta}(T)+\int^T_0 \frac{d\lambda}{dt}\frac{dx}{d\theta}dt$ 
(由于$x(0)=0\Rightarrow\frac{dx}{d\theta}(0)=0$)

为使$\frac{dx}{d\theta}$的系数为0，得到关于λ的微分方程：
$\frac{\partial g}{\partial x} + \lambda^TA+\dot{\lambda}^T=0, \lambda(T)=0$
其中$\frac{\partial g}{\partial x}$可在已知ground truth时，对由predict计算loss的过程取Jacobian获得.
令$P=(\frac{\partial g}{\partial x})^T$,上述方程两边取转置，可化为：
$\dot\lambda+A^T\lambda+P=0, \lambda(T)=0$
对该方程使用RK4求解即可得到伴随乘子$\lambda(t)$.

现在，我们要求的目标梯度为：
$\frac{dJ}{d\theta}=\frac{dL}{d\theta}=\int[\lambda(t)^T\frac{\partial f}{\partial \theta}]dt$
其中$\frac{\partial f}{\partial \theta}(t)_{n*n_\theta}=\frac{\partial f}{\partial A}(t)_{n*(n*n)}\frac{\partial A}{\partial \theta}_{(n*n)*n_\theta}+\frac{\partial f}{\partial b}\frac{\partial b}{\partial \theta}$
$\frac{\partial A}{\partial \theta}$与$\frac{\partial b}{\partial \theta}$为定值，可预计算；
$\frac{\partial f}{\partial A}(t)$=($x(t)$.unsqueeze(-1).unsqueeze(-1)*torch.eye(n)).premute(1, 2, 0).
$\frac{\partial f}{\partial b}=1$

总而言之，上述计算只需要存储每个时刻的x(t)与预计算梯度$\frac{\partial A}{\partial \theta}$,其空间复杂度为$O(n_tn+n_\theta n^2)$,远低于梯度反传方法记录每个矩阵每个时刻梯度所需$O(n_tn^2)$。

#### 与model reduction的结合
model reduction:
$K_0U = M_0US$
其中$M_0, K_0$是原始质量矩阵与刚度矩阵，U是广义特征向量矩阵,S是特征值对角矩阵,满足
$U^T M_0U = I, U^TK_0U = S$
对$u=Uq$，
振动方程由
$M_0\ddot u + C_0\dot u + K_0u = f(t)$
化为：
$\ddot q + (αI + βS)\dot q + Sq = U^T f(t)$.
此时有：
$M=I,C=αI + βS, K=S$
$\frac{\partial M}{\partial \theta}=0,\frac{\partial C}{\partial \theta}=\beta \frac{\partial S}{\partial \theta},
\frac{\partial K}{\partial \theta}=\frac{\partial S}{\partial \theta}$

关于$\frac{\partial S}{\partial \theta}$的计算，注意到我们原来处理特征值的梯度的方法：
$\frac{\partial \lambda_i}{\partial \theta} = u_i^T(\frac{\partial K_0}{\partial \theta}-\lambda_i\frac{\partial M_0}{\partial \theta})u_i$
即：
$\frac{\partial S}{\partial \theta} = (U^T\frac{\partial K_0}{\partial \theta}U)-(U^T\frac{\partial M_0}{\partial \theta}U)S$
问题：上面一步推导要求$U^T\frac{\partial K_0}{\partial \theta}U和U^T\frac{\partial M_0}{\partial \theta}U$是对角矩阵，是这样吗？




