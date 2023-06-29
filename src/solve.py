import torch
import warnings
from tqdm import tqdm


class BiCGSTAB():
    """
    modified from https://gist.github.com/bridgesign/f421f69ad4a3858430e5e235bccde8c6
    This is a pytorch implementation of BiCGSTAB solver.
    """

    def __init__(self, A, preconditioner=None, device='cuda'):
        self.A = A
        self.preconditioner = preconditioner
        self.device = device

    def matvec(self, x):
        if self.preconditioner is None:
            return self.A(x)
        else:
            return self.preconditioner(self.A(x))

    def init_params(self, b, x=None, max_iter=None, tol=1e-10, atol=1e-16):
        """
        b: The R.H.S of the system. 1-D tensor
        max_iter: Number of steps of calculation
        tol: Tolerance such that if ||r||^2 < tol * ||b||^2 then converged
        atol:  Tolernace such that if ||r||^2 < atol then converged
        """
        self.b = b
        self.x = torch.zeros(
            b.shape[0], device=self.device) if x is None else x
        self.residual_tol = tol * torch.vdot(b, b).item()
        self.atol = torch.tensor(atol, device=self.device)
        self.max_iter = b.shape[0] if max_iter is None else max_iter
        self.status, self.r = self.check_convergence(self.x)
        self.rho = torch.tensor(1, device=self.device)
        self.alpha = torch.tensor(1, device=self.device)
        self.omega = torch.tensor(1, device=self.device)
        self.v = torch.zeros(b.shape[0], device=self.device)
        self.p = torch.zeros(b.shape[0], device=self.device)
        self.r_hat = self.r.clone().detach()

    def check_convergence(self, x):
        r = self.b - self.matvec(x)
        # print("r", r)
        rdotr = torch.vdot(r, r).real
        if rdotr < self.residual_tol or rdotr < self.atol:
            return True, r
        else:
            return False, r

    def step(self):
        # rho_i <- <r0, r^>
        rho = torch.dot(self.r, self.r_hat)
        # beta <- (rho_i/rho_{i-1}) x (alpha/omega_{i-1})
        beta = (rho / self.rho) * (self.alpha / self.omega)
        # rho_{i-1} <- rho_i  replaced self value
        self.rho = rho
        # p_i <- r_{i-1} + beta x (p_{i-1} - w_{i-1} v_{i-1}) replaced p self value
        self.p = self.r + beta * (self.p - self.omega * self.v)
        self.v = self.matvec(self.p)                            # v_i <- Ap_i
        # alpha <- rho_i/<r^, v_i>
        self.alpha = self.rho / torch.dot(self.r_hat, self.v)
        # h_i <- x_{i-1} + alpha p_i
        self.h = self.x + self.alpha * self.p
        status, res = self.check_convergence(self.h)
        if status:
            self.x = self.h
            return True
        # s <- r_{i-1} - alpha v_i
        s = self.r - self.alpha * self.v
        t = self.matvec(s)                                     # t <- As
        # w_i <- <t, s>/<t, t>
        self.omega = torch.dot(t, s) / torch.dot(t, t)
        # x_i <- x_{i-1} + alpha p + w_i s
        self.x = self.h + self.omega * s
        status, res = self.check_convergence(self.x)
        if status:
            return True
        else:
            self.r = s - self.omega * t                           # r_i <- s - w_i t
            return False

    def solve(self, b, x=None, max_iter=None, tol=1e-10, atol=1e-16):
        """
        Method to find the solution.
        Returns the final answer of x
        """
        if self.preconditioner is not None:
            b = self.preconditioner(b)
        # print('Solving the system...')
        # print('b:', b)
        self.init_params(b, x, max_iter, tol, atol)
        if self.status:
            return self.x
        while self.max_iter:
            s = self.step()
            if s:
                return self.x
            if self.rho == 0:
                break
            self.max_iter -= 1
        warnings.warn('Convergence has failed :(')
        return self.x


class EyeSolver():
    def __init__(self):
        pass

    def solve(self, b):
        return b


class WaveSolver():

    def __init__(self, mass_matrix, damping_matrix, stiffness_matrix, force, dt, batch_size=1):
        '''
        mass_matrix: mass matrix, a sparse matrix|'identity'
        damping_matrix: damping matrix, a function of v
        stiffness_matrix: stiffness matrix, a function of x
        force: force, a function of time
        dt: time step
        '''
        self.C = damping_matrix
        self.K = stiffness_matrix
        self.force = force
        self.dt = dt
        self.batch_size = batch_size
        if mass_matrix == 'identity':
            self.M_mat = torch.eye(force(0).shape[-1]).cuda()
            self.mass_linear_solver = EyeSolver()
        else:
            self.M_mat = mass_matrix
            self.mass_linear_solver = self.init_mass_linear_solver()
            
    def update(self, mass_matrix=None, damping_matrix=None, stiff_matrix=None):
        if mass_matrix is not None:
            self.M_mat = mass_matrix
            self.mass_linear_solver = self.init_mass_linear_solver()
        if damping_matrix is not None:
            self.C = damping_matrix
        if stiff_matrix is not None:
            self.K = stiff_matrix
            

    def init_mass_linear_solver(self):
        # preconditioner as inverse of diagonal of mass matrix (which is a sparse matrix)
        rows = self.M_mat.indices()[0]
        cols = self.M_mat.indices()[1]
        values = self.M_mat.values()
        diag_rows = rows[rows == cols]
        diag_values = values[rows == cols]
        # resorted diag_values using diag_rows
        diag_values = diag_values[torch.argsort(diag_rows)]
        self.mass_matrix_diag = diag_values

        def matvec(v):
            return self.M_mat @ v

        def preconditioner(v):
            return v / diag_values

        return BiCGSTAB(matvec, preconditioner)

    def derivative(self, v, x, t):
        f = self.force(t)
        c = self.C(v)
        
        # here, in fact the input x and output k are reduced low-dim, but the calculation of k(x) are in a high-dim.
        # can we save the jacobian matrix of k(x) and backward only use it?
        # Oh no! the output of network is the weight of lame_mu and lame_lambda, which is used in k(x).
        k = self.K(x) 
         
        rhs = f - c - k
        a = self.mass_linear_solver.solve(rhs)
        return a, v

    def solve(self, num_step):
        '''
        solve the linear system using the RK4 method
        '''

        # initial condition
        v = torch.zeros([self.batch_size, self.M_mat.shape[0]], dtype=torch.float64, requires_grad=True).to(
            self.M_mat.device)
        x = torch.zeros([self.batch_size, self.M_mat.shape[0]], dtype=torch.float64, requires_grad=True).to(
            self.M_mat.device)

        xs = []

        for i in tqdm(range(num_step)):
            t = i * self.dt
            k1, l1 = self.derivative(v, x, t)
            k2, l2 = self.derivative(
                v + self.dt / 2 * k1, x + self.dt / 2 * l1, t + self.dt / 2)
            k3, l3 = self.derivative(
                v + self.dt / 2 * k2, x + self.dt / 2 * l2, t + self.dt / 2)
            k4, l4 = self.derivative(
                v + self.dt * k3, x + self.dt * l3, t + self.dt)

            v = v + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            x = x + self.dt / 6 * (l1 + 2 * l2 + 2 * l3 + l4)

            xs.append(x)

        return torch.stack(xs, dim=0)


class ModalWaveSolver():

    def __init__(self, derivative_func, mode_num, dt):
        '''
        derivative_func: a function of v, x, t
        mode_num: number of modes
        dt: time step
        '''
        self.derivative = derivative_func
        self.mode_num = mode_num
        self.dt = dt

    def solve(self, num_step):
        '''
        solve the linear system using the RK4 method
        '''

        # initial condition
        v = torch.zeros(self.mode_num, requires_grad=True).cuda()
        x = torch.zeros(self.mode_num, requires_grad=True).cuda()
        xs = []

        for i in range(num_step):
            t = i * self.dt
            k1, l1 = self.derivative(v, x, t)
            k2, l2 = self.derivative(
                v + self.dt / 2 * k1, x + self.dt / 2 * l1, t + self.dt / 2)
            k3, l3 = self.derivative(
                v + self.dt / 2 * k2, x + self.dt / 2 * l2, t + self.dt / 2)
            k4, l4 = self.derivative(
                v + self.dt * k3, x + self.dt * l3, t + self.dt)

            v = v + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            x = x + self.dt / 6 * (l1 + 2 * l2 + 2 * l3 + l4)
            xs.append(x)

        return torch.stack(xs, dim=0)
