import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
import pandas as pd


def plot2D(N, Z, title=""):
    X,Y = np.ogrid[0:1:(N+1)*1j, 0:1:(N+1)*1j]
    # Define a new figure with given size
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z,             
                           rstride=1, cstride=1, # Sampling rates for the x and y input data
                           cmap=cm.viridis)      # Use the new fancy colormap viridis
    # Set initial view angle
    ax.view_init(30, 225)
    
    # Set labels and show figure
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(title)
    plt.show()



def Lh(M):
    """
    Dicretized laplacian operator in R^2 (d_xx + d_yy) working on internal entries of M
    """
    C = M.copy()
    N = C.shape[0]-1
    
    index = np.arange(1,N)
    shift = lambda s: np.ix_(index + s[0],index + s[1])
    s = np.array([[-1,0],[1,0],[0,-1],[0,1],[0,0]])

    w = ((N)**2)*np.array([1,1,1,1,-4])
    D = np.zeros((N-1,N-1))
    for i in range(len(w)):
        D += w[i]*C[shift(s[i])]
    C[shift([0,0])] = D
    return C


def zero_boundary(A):
    """
    Setting the boundary entries of matrix A to zero
    """
    n = A.shape[0]
    Z = Z = np.zeros((n,n))
    Z[1:n-1,1:n-1] = A[1:n-1,1:n-1]
    return Z


def residual(u,rhs):
    """
    Finding the residual of the problem Au = rhs, 
    assuming residual zero on boundaries
    """
    
    r = Lh(u) + rhs
    r = zero_boundary(r)
    return r



def cg(u0,rhs,N,tol,max_iter):
    
    """
    Conjugate gradient algorithm solving Au = rhs
    
    INPUT
        u0: initial guess, numpy array (N+1) x (N+1)
        rhs: right hand side of problem Au = rhs, numpy array (N+1) x (N+1)
        N: Size of problem, integer
        tol: Tolerance defining convergence criteria: assume convergence when relative residual < tol
        max_iter: Threshold for maximum number of iterations if convergence criteria is not met
        
    OUTPUT
        uj: approximate soltion of Au = rhs
        res_list: list of the residuals for each iteration
        time_delta: number of seconds it took to reach convergence

    """
    
    r0 = residual(u0,rhs)
    res_list = []
    
    rres = 1
    rj = r0
    pj = r0
    uj = u0
    it = 0

    time_start = time.time()
    
    while rres > tol and it < max_iter:
        pj = zero_boundary(pj)
        Apj = -Lh(pj)
        
        eta = np.sum(rj*rj)/np.sum(Apj*pj)
        ujp = uj + eta*pj
        rjp = rj - eta*Apj
        delta = np.sum(rjp*rjp)/np.sum(rj*rj)
        pjp = rjp + delta*pj
        
        it += 1
        
        rres = np.linalg.norm(rj,ord='fro')/np.linalg.norm(r0,ord='fro')
        res_list.append(np.linalg.norm(rj,ord='fro'))
        rj,pj,uj = rjp,pjp,ujp
        
    time_end = time.time()
    time_delta = time_end - time_start
    
    return uj,res_list,time_delta



def test_cg(N_min,n_tests,tol):
    
    """
    Testing the Conjugate gradient algorithm on the poisson problem
    with homogenous boundary conditions
    
    INPUT
        N_min: Size of smallest problem (Au = rhs, meaning A is a matrix of size (N+1) x (N+1))
        n_test: Number of tests to be conducted. 
        Problem size increases with a factor of 2 for each test.
        
        tol: Tolerance defining the convergence criteria in CG. 
        
    OUTPUT
        u: Approximate solution of the final test problem (largest)
        N: Size of final test problem
        table: Pandas dataframe with iteration number, problem size and time to convergence for each test
    
    """
    
    run_time = [0]*n_tests
    iter_list = [0]*n_tests
    N_list = [0]*n_tests
    
    
    def get_problem_homogenous(N):
        u0 = np.random.rand(N+1,N+1)
        x,y = np.ogrid[0:1:(N+1)*1j, 0:1:(N+1)*1j]
        f = lambda x,y: 20*np.pi**2*np.sin(2*np.pi*x)*np.sin(4*np.pi*y)
        g = lambda x,y: np.sin(2*np.pi*x)*np.sin(4*np.pi*y)
        z = g(x,y)
        z[1:N,1:N] = u0[1:N,1:N]
        u0 = z
        rhs = f(x,y)
        return u0,rhs
    

    x = lambda l: np.arange(len(l))
    for i in range(n_tests):
        N = N_min*2**i
        u0,rhs = get_problem_homogenous(N)
        u,res_list,run_time[i] = cg(u0,rhs,N,tol,1000)
        N_list[i] = N
        iter_list[i] = len(res_list)
        
        plt.semilogy(x(res_list),res_list,label="N = " + str(N))
        
    table = pd.DataFrame({"Problem size N":N_list,
        "Iterations":iter_list,
        "Total run time":run_time}
        )

        
    plt.title("Convergence of CG")
    plt.xlabel("Iteration number, k")
    plt.ylabel("Residual " + r'$||r_k\||_2$')
    plt.legend()
    plt.show()
    
    return u,N,table



def jacobi(u0,rhs,omega,nu):
    
    """
    Implementation of the weighted Jacobi algorithm.
    
    INPUT
        u0: initial guess, numpy array (N+1) x (N+1)
        rhs: right hand side of problem Au = rhs, numpy array (N+1) x (N+1)
        omega: relaxation parameter, how much of next Jacobi iteration should be "taken"
        nu: Number of iterations
        
    OUTPUT
        u0: approximate solution of Au = rhs
    
    
    """
    
    
    def Jh(u,rhs):
        
        #One jacobi iteration. Works directly on the grid.
        
        C = u.copy()
        N = C.shape[0]-1
        index = np.arange(1,N)
        shift = lambda s: np.ix_(index + s[0],index + s[1])
        s = np.array([[-1,0],[1,0],[0,-1],[0,1]])
        D = np.zeros((N-1,N-1))
        for i in range(len(s)):
            D += 0.25*C[shift(s[i])]
        D += 0.25*(1/(N))**2*rhs[shift([0,0])]
        C[shift([0,0])] = D
        return C
        
    for i in range(nu):
        u1 = (1-omega)*u0 + omega*Jh(u0,rhs)
        u0 = u1.copy()
    return u0



def resize_grid(A,grid_scaling):
    
    """
    Restriction and interpolation function for the Multigrid algorithm.
    
    INPUT
        A: numpy array (N+1) x (N+1) to be resized
        grid_scaling: string either "restrict" or "interpolate" choosing which operation to be taken
        
    OUTPUT
        B: the restricted or interpolated numpy array
    
    """
    
    shift = lambda s,index: np.ix_(index + s[0],index + s[1])
    s = np.array([[-1,0],[1,0],[0,-1],[0,1],[0,0],[1,1],[-1,-1],[1,-1],[-1,1]])
    w = (1/16)*np.array([2]*4+[4]+[1]*4)
    N = A.shape[0]

    if grid_scaling == "restrict":
        C = A.copy()
        n = N//2
        index_all = np.arange(N)[0:n+1]*2
        index_inner = index_all[1:n]
        C_reduced = A[np.ix_(index_all,index_all)] 
        
        B = np.zeros((n-1,n-1))
        for i in range(len(s)):
            B += w[i]*A[shift(s[i],index_inner)]
        C_reduced[np.ix_(index_inner//2,index_inner//2)] = B
        B = C_reduced

    elif grid_scaling == "interpolate":
        w = w*4
        n = N*2-1
        index = np.arange(n)
        index_coarse = np.arange(n)[0:n//2+1]*2 + 1
        B = np.zeros((n+2,n+2))
        for i in range(len(s)):
            B[shift(s[i],index_coarse)] += A*w[i]
        C = B.copy()[1:n+1,1:n+1]
        B = C
     
    return B

def mgv(u0,rhs,N,nu1,nu2,level,max_level):
    
    """
    The function mgv(u0,rhs,N,nu1,nu2,level,max_level) performs
    one multigrid V-cycle on the 2D Poisson problem on the unit
    square [0,1]x[0,1] with initial guess u0 and righthand side rhs.
    
    INPUT 
        u0: initial guess
        rhs: righthand side
        N: u0 is a (N+1)x(N+1) matrix
        nu1: number of presmoothings
        nu2: number of postsmoothings
        level: current level
        max_level: total number of levels
        
    OUTPUT
        u: approximate solution of the problem Au = rhs
    """
    
    
    if level == max_level:
        u,res_list,time_delta = cg(u0,rhs,N,1e-13,1000)
    else:
        u = jacobi(u0,rhs,2/3,nu1)
        rf = residual(u,rhs)
        rc = resize_grid(rf,"restrict")
        ec = mgv(np.zeros((int(N/2)+1,int(N/2)+1)),rc,int(N/2),nu1,nu2,level+1,max_level)
        ef = resize_grid(ec,"interpolate")
        u = u + ef
        u = jacobi(u,rhs,2/3,nu2)
    return u



def mgv_iterations(u0,rhs,N,nu1,nu2,level,max_level,plot5,tol):
    
    """
    Multigrid iterations: Uses the mgv(u0,rhs,N,nu1,nu2,level,max_level) function 
    and performs iterations on the problem Au = rhs until convergence
    
    INPUT
        u0,rhs,N,nu1,nu2,level,max_level: See help(mgv)
        plot5: plots the five first approximate solutions u when N == 32
        tol: Tolerance defining convergence criteria: assume convergence when relative residual < tol

    OUTPUT
        u: approximate soltion of the problem Au = rhs
        res_list: list of the residuals for each iteration
        time_delta: number of seconds it took to reach convergence
        
    
    """
    
    it = 0
    r0 = residual(u0,rhs)
    res_list = [np.linalg.norm(r0,ord='fro')]
    rres = 1
    
    time_start = time.time()
    
    while rres > tol:
        u = mgv(u0,rhs,N,nu1,nu2,level,max_level)
        r = residual(u,rhs)
        u0 = u
        rnorm = np.linalg.norm(r,ord='fro')
        res_list.append(rnorm)
        rres = rnorm/np.linalg.norm(r0,ord='fro')
        
        if plot5 and len(res_list)-1 < 6:
            plot2D(N,u,title= "MGV Solution after " + str(len(res_list)-1) + " iterations")
    
    time_end = time.time()
    time_delta = time_end - time_start
    
    return u,res_list,time_delta



def get_problem_non_homog(N):
    
    """
    Get discretized variables defining the 2D poission problem
    with non homogenous Dirichlet boundary conditions.
    
    INPUT
        N: Size of problem
        
    OUTPUT
        u0: initial guess, a numpy array (N+1) x (N+1) with random floats (0,1)
        rhs: right hand side of problem Au = rhs, numpy array (N+1) x (N+1)
    
    """
    
    x,y = np.ogrid[0:1:(N+1)*1j, 0:1:(N+1)*1j]
    f = lambda x,y: -1 + x*y*0
    def get_bc(N):
        C = np.zeros((N+1,N+1))
        g = lambda x,y: 4*y*(1-y) + x*0
        D = g(x,y)
        C[0,:] = D[0,:]
        return C
    bc = get_bc(N)
    u0 = np.random.rand(N+1,N+1)
    bc[1:N,1:N] = u0[1:N,1:N]
    u0 = bc
    rhs = f(x,y)
    return u0,rhs



def test_mgv(N_min,n_tests,nu1,nu2,tol,plot5=True): 
    
    """
    Testing the Multigrid algorithm on the 2D poisson problem
    with non-homogenous boundary conditions
    
    INPUT
        N_min: Size of smallest problem (Au = rhs, meaning A is a matrix of size (N+1) x (N+1))
        n_test: Number of tests to be conducted. Problem size increases with a factor of 2 for each test.
        nu1: number of presmoothings
        nu2: number of postsmoothings
        
        
    OUTPUT
        u: Approximate solution of the final test problem (largest)
        u0: Initial guess for the final test problem
        N: Size of final test problem
        Table: Pandas dataframe with iteration number, problem size and time to convergence for each test
    
    """
    
    run_time = [0]*n_tests
    iter_list = [0]*n_tests
    N_list = [0]*n_tests

        
    x = lambda l: np.arange(len(l))
    
    for i in range(n_tests):
        N = N_min*2**i
        if N == 2**5 and plot5:
            plot5 = True
        else:
            plot5 = False
        max_level = i+2
        u0,rhs = get_problem_non_homog(N)
        u,res_list,run_time[i] = mgv_iterations(u0,rhs,N,nu1,nu2,0,max_level,plot5,tol)
        iter_list[i] = len(res_list)
        N_list[i] = N
        plt.semilogy(x(res_list),res_list,label="N = " + str(N))

    plt.title("Convergence of MGV")
    plt.xlabel("Iteration number, k")
    plt.ylabel("Residual " + r'$||r_k\||_2$')
    plt.legend()
    plt.show()
    
    table = pd.DataFrame({"Problem size N":N_list,
            "Iterations":iter_list,
            "Total run time":run_time}
            )

    return u,u0,N,table



def pcg(u0,rhs,N,tol,nu1,nu2,max_iter,max_level,plot5):
    
    """
    Algorithm solving the 2D Poisson problem using
    a preconditioned conjugate gradient algorithm.
    Uses a multigrid V-cycle scheme as preconditioner.
    
    INPUT
        u0,rhs,N,nu1,nu2,level,max_level: See help(mgv)
        plot5: plots the five first approximate solutions u when N == 32
    
    OUTPUT
        u: approximate soltion of the problem Au = rhs
        res_list: list of the residuals for each iteration
        time_delta: number of seconds it took to reach convergence
    
    """
    
    
    r0 = residual(u0,rhs)
    rres = 1
    res_list = []
    rj = r0
    zj = mgv(np.zeros((N+1,N+1)),rj,N,nu1,nu2,1,max_level)
    x,y = np.ogrid[0:1:(N+1)*1j, 0:1:(N+1)*1j]
    
    pj = zj
    uj = u0
    it = 0
    
    time_start = time.time()
    
    while rres > tol and it < max_iter:
        
        Z = np.zeros((N+1,N+1))
        Z[1:N,1:N] = pj[1:N,1:N]
        pj = Z.copy()
        
        Apj = -Lh(pj)
        eta = np.sum(zj*rj)/np.sum(Apj*pj)
        ujp = uj + eta*pj
        rjp = rj - eta*Apj
        zjp = mgv(np.zeros((N+1,N+1)),rjp,N,nu1,nu2,1,max_level)
        delta = np.sum(zjp*rjp)/np.sum(zj*rj)
        pjp = zjp + delta*pj
        
        it += 1
        if plot5 and it < 6:
            plot2D(N,ujp,title= "PCG Solution after " + str(it) + " iterations")


        rnorm = np.linalg.norm(rjp,ord='fro')
        res_list.append(rnorm)
        rres = rnorm/np.linalg.norm(r0,ord='fro')
        
        rj,pj,uj,zj = rjp,pjp,ujp,zjp
        
    time_end = time.time()
    time_delta = time_end - time_start
        
    return uj,res_list,time_delta


def test_pcg(N_min,n_tests,nu1,nu2,tol,plot5=True): 
    
    
    """
    Testing the preconditioned conjugate gradient (PCG) algorithm 
    on the 2D poisson problem with non-homogenous boundary conditions
    
    INPUT
        N_min: Size of smallest problem (Au = rhs, meaning A is a matrix of size (N+1) x (N+1))
        n_test: Number of tests to be conducted. Problem size increases with a factor of 2 for each test.
        nu1: number of presmoothings
        nu2: number of postsmoothings
        
        
    OUTPUT
        u: Approximate solution of the final test problem (largest)
        u0: Initial guess for the final test problem
        N: Size of final test problem
        Table: Pandas dataframe with iteration number, problem size and time to convergence for each test
    
    """
    
    
    x = lambda l: np.arange(len(l))
    max_iter = 1000
    run_time = [0]*n_tests
    iter_list = [0]*n_tests
    N_list = [0]*n_tests
    
    
    for i in range(n_tests):
        N = N_min*2**i
        
        if N == 2**5 and plot5:
            plot5 = True
        else:
            plot5 = False
        max_level = i+2
        u0,rhs = get_problem_non_homog(N)
        u,res_list,run_time[i] = pcg(u0,rhs,N,tol,nu1,nu2,max_iter,max_level,plot5)
        iter_list[i] = len(res_list)
        N_list[i] = N
        plt.semilogy(x(res_list),res_list,label="N = " + str(N))

    plt.title("Convergence of PCG")
    plt.xlabel("Iteration number, k")
    plt.ylabel("Residual " + r'$||r_k\||_2$')
    plt.legend()
    plt.show()
    
    table = pd.DataFrame({"Problem size N":N_list,
                        "Iterations":iter_list,
                        "Total run time":run_time}
                        )
    
    return u,u0,N,table