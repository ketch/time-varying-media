from clawpack import pyclaw
import numpy as np
from clawpack import riemann

ZA = 1.
ZB = 1.1
cA = 1.
cB = np.pi/2.
alpha = 0.5

def setaux(x,t):
    aux = np.empty([2,len(x)],order='F')
    tfrac = t-np.floor(t)
    # Impedance:
    aux[0,:] = ZA*(tfrac<alpha)+ZB*(tfrac>=alpha)
    #Bulk modulus:
    aux[1,:] = cA*(tfrac<alpha)+cB*(tfrac>=alpha)

    return aux

def update_aux(solver,state):
    new_aux = setaux(state.grid.x.centers,state.t)
    state.aux[:] = new_aux


def setup(tfinal=20.,num_output_times=100, solver_type="classic"):
    riemann_solver = riemann.acoustics_variable_1D
    if solver_type == "classic":
        solver = pyclaw.ClawSolver1D(riemann_solver)
    elif solver_type == "sharpclaw":
        solver = pyclaw.SharpClawSolver1D(riemann_solver)
        solver.call_before_step_each_stage = True

    num_cells = 20000
    x = pyclaw.Dimension(-50.0, 50.0, num_cells, name='x')
    domain = pyclaw.Domain(x)
    num_eqn = 2; num_aux=2
    state = pyclaw.State(domain, num_eqn, num_aux)
    solver.bc_lower[0] = pyclaw.BC.periodic
    solver.bc_upper[0] = pyclaw.BC.periodic
    solver.aux_bc_lower[0] = pyclaw.BC.periodic
    solver.aux_bc_upper[0] = pyclaw.BC.periodic

    solver.before_step = update_aux

    xc = domain.grid.x.centers
    state.aux[0,:] = 1.#*(xc<0.) + 2.*(xc>=0)  # impedance
    state.aux[1,:] = 1.  # sound speed
    state.q[1,:] = np.exp(-(xc/5)**2)
    state.q[0,:] = state.aux[0,:] * state.q[1,:]  # right-going wave

    claw = pyclaw.Controller()
    claw.solution = pyclaw.Solution(state, domain)
    claw.solver = solver
    claw.keep_copy=True
    #claw.output_format=None
    claw.tfinal = tfinal
    claw.num_output_times = num_output_times

    return claw

if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup)
