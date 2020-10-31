from multigrid import plot2D

#size of the first test problems
N_start = 32

#Number of test problems to run. Each test problem doubles in size.
n_tests = 4

#Tolerance defining convergence criteria measured in the relative residual.
tol = 1e-12

#Number of presmoothings by the Jacobi method in mgv and pcg
nu1 = 2

#Number of postsmoothings by the Jacobi method in mgv and pcg
nu2 = 2

#Plots the largest solution and initial guess for each method
plotU = True

#Plots the solution for the five first iteerations for mgv and pcg
plot5 = False


# ## 1. Conjugate gradient
# 
# Testing convergence of the conjugate gradient algorithm for various problem sizes.


from multigrid import test_cg
u,N,table_cg = test_cg(N_start,n_tests,tol)

if plotU:
    plot2D(N,u,title = "CG Solution of Poission problem, N = " + str(N))


# ## 2. Multigrid V-cycle
# 
# Testing convergence of multiple iterations with the multigrid V-cyle algorithm for various problem sizes.



from multigrid import test_mgv
u,u0,N,table_mgv = test_mgv(N_start,n_tests,nu1,nu2,tol,plot5)


if plotU:
    plot2D(N,u0,title= "MGV Initial guess, N = " + str(N))
    plot2D(N,u,title = "MGV Solution of Poission problem, N = " + str(N))

# ## 3. Multigrid preconditioner for conjugate gradient
# 
# Testing convergence of multiple iterations using the multigrid method as a preconditioner for conjugate gradient. 


from multigrid import test_pcg
u,u0,N,table_pcg = test_pcg(N_start,n_tests,nu1,nu2,tol,plot5)


if plotU:
    plot2D(N,u0,title= "PCG Initial guess, N = " + str(N))
    plot2D(N,u,title = "PCG Solution of Poission problem, N = " + str(N))


# ## 4. Convergence metrics
# 
# Below you will find tables allowing for a comparison of number of iterations and the computational time for the different algorithms. Notice in particular the superiority of the preconditioned conjugate gradient algorithm both in terms of time and number of iterations. 



print("Conjugate gradient")
display(table_cg)
print("Multigrid")
display(table_mgv)
print("Preconditioned conjugate gradient")
display(table_pcg)

