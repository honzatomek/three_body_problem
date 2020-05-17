#!/usr/bin/python3

# Import scipy
import scipy as sci
import scipy.integrate
# Import matplotlib and associated modules for 3D and animations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
# from matplotlib import animation
import matplotlib.animation as animation

# <-------------------------------------------------------------------------- global variabbles --->
ANIMATION_LENGTH_SEC = 10

class OneBody:
    def __init__(self, name, mass, position, velocity, color):
        self.name = name
        self.mass = mass
        self.position = sci.array(position, dtype="float64")
        self.velocity = sci.array(velocity, dtype="float64")
        self.color = color

    def __str__(self):
        retval = '\t{0:<19}:'.format(self.name)
        for i, d in enumerate(['x', 'y', 'z']):
            retval += '\t{0:<2} = {1: 8.3f}'.format(d, self.position[i])
        retval += '\n\t' + 20 * ' '
        for i, d in enumerate(['vx', 'vy', 'vz']):
            retval += '\t{0:<2} = {1: 8.3f}'.format(d, self.velocity[i])
        retval += '\n\t' + 20 * ' '
        retval += '\t{0:<2} = {1: 8.3f}'.format('m', self.mass)
        
        return retval


class MultiBodyProblem:

    def __init__(self, periods=8, points=500):
        print('[\033[01;32m+\033[0m] Initialising MultiBody Problem solver.\n    periods: {0}\tintegration points: {1}'.format(periods, points))
        self.periods = periods
        self.points = points

        # Define universal gravitation constant
        self.G = 5.67408e-11  # N-m2/kg2
        # Reference quantities
        self.m_nd = 0.989e+30  # kg #mass of the sun
        self.r_nd = 4.326e+12  # m #distance between stars in Alpha Centauri
        self.v_nd = 29999  # m/s #relative velocity of earth around the sun
        self.t_nd = 78.91 * 365 * 24 * 3600 * 0.51  # s #orbital period of Alpha Centauri
        # Net constants
        self.K0= self.G * self.t_nd * self.m_nd / (self.r_nd ** 2 * self.v_nd)
        self.K1 = self.v_nd * self.t_nd / self.r_nd

        self.bodies = []

        self.COM = None

        self.init_params = None
        self.time_span = None

    def add_body(self, mass, position, velocity, name=None, color=None):
        if name is None:
            name = 'Star {0}'.format(len(self.bodies) + 1)
        self.bodies.append(OneBody(name, mass, position, velocity, color))
        print('[\033[01;32m+\033[0m] Added Celestial Body:\n{0}'.format(str(self.bodies[-1])))
        self.init_COM()

    def init_COM(self):
        r_numerator = 0
        v_numerator = 0
        denominator = 0
        for body in self.bodies:
            r_numerator += body.mass * body.position
            v_numerator += body.mass * body.velocity
            denominator += body.mass

        self.COM = OneBody('COM', denominator, r_numerator / denominator, v_numerator / denominator, "tab:yellow")

    #A function defining the equations of motion
    def MultiBodyEquations(self, w, t):  # , G):
        r = []
        v = []
        num = len(self.bodies)
        for i in range(num):
            r.append(w[i * 3: (i + 1) * 3])
            v.append(w[(i + num) * 3: (i + num + 1) * 3])

        dv_dt = []
        dr_dt = []
        for i in range(num):
            dv_dt.append(0)
            dr_dt.append(self.K1 * v[i])
            for j in range(num):
                if i != j:
                    dv_dt[i] += self.K0 * self.bodies[j].mass * (r[j] - r[i]) / (sci.linalg.norm(r[j] - r[i]) ** 3)

        r_derivs = sci.concatenate((dr_dt[0], dr_dt[1]))
        v_derivs = sci.concatenate((dv_dt[0], dv_dt[1]))
        if num > 2:
            for i in range(2, num):
                r_derivs = sci.concatenate((r_derivs, dr_dt[i]))
                v_derivs = sci.concatenate((v_derivs, dv_dt[i]))
        derivs = sci.concatenate((r_derivs, v_derivs))
        return derivs

    def initialize(self):
        self.init_params = []
        for body in self.bodies:
            self.init_params.append(body.position)
        for body in self.bodies:
            self.init_params.append(body.velocity)

        self.init_params = sci.array(self.init_params).flatten()
        self.time_span = sci.linspace(0, self.periods, self.points)

    def solve(self, relative_to_com=False):
        print('[\033[01;32m+\033[0m] Running solver.')
        multi_body_solution = sci.integrate.odeint(self.MultiBodyEquations,
                                                   self.init_params,
                                                   self.time_span)
        r_sol = []
        limits = [[multi_body_solution.max(), multi_body_solution.min()],
                  [multi_body_solution.max(), multi_body_solution.min()],
                  [multi_body_solution.max(), multi_body_solution.min()]]

        for i, body in enumerate(self.bodies):
            r_sol.append(sci.transpose(multi_body_solution[:, i*3:(i + 1) * 3]))
            for j, tmp in enumerate(sci.transpose(multi_body_solution[:, i*3:(i + 1) * 3])):
                limits[j] = [min(limits[j][0], tmp.min()), max(limits[j][1], tmp.max())]

        if relative_to_com:
            rcom_sol = self.bodies[0].mass * r_sol[0]
            m = self.bodies[0].mass
            for i, body in enumerate(self.bodies[1:]):
                rcom_sol += body.mass * r_sol[i + 1]
                m += body.mass
            rcom_sol = rcom_sol / m
            limits = [[rcom_sol.max(), rcom_sol.min()],
                      [rcom_sol.max(), rcom_sol.min()],
                      [rcom_sol.max(), rcom_sol.min()]]
            for i, sol in enumerate(r_sol):
                sol -= rcom_sol
                for j in range(3):
                    limits[j] = [min(limits[j][0], sol[j].min()), max(limits[j][1], sol[j].max())]

        print('[\033[01;32m+\033[0m] Output data relative to Center of Mass: {0}.'.format(str(relative_to_com)))
        return r_sol, limits


def update_lines(num, dataLines, lines, dots):
    for line, data in zip(lines, dataLines):
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
    
    for dot, data in zip(dots, dataLines):
        # dot._offsets3d = [float(data[0, num]), float(data[1, num]), float(data[2, num])]
        dot.set_data(data[0:2, num])
        dot.set_3d_properties(data[2, num])

    return lines, dots


if __name__ == '__main__':
    # Create figure
    print('[\033[01;32m+\033[0m] Started script to plot Multi-Body Problem using ODE.')
    fig = plt.figure(figsize=(9, 9))
    ax = p3.Axes3D(fig)
    
    # initialize class
    mbp = MultiBodyProblem(20, 1500)
    
    # set up the celestial bodies
    mbp.add_body(1.1, [-0.5, 0.0, 0.0], [0.01, 0.01, 0.0], 'Alpha Centauri A', "tab:blue")
    mbp.add_body(0.907, [0.5 , 0.0, 0.0], [-0.05, 0.0, -0.1], 'Alpha Centauri B', "tab:red")
    mbp.add_body(0.985, [0.0, 1.0, 0.0], [0.0, -0.005, 0.0], 'Alpha Centauri C', "tab:purple")
    # mbp.add_body(1.0, [0.0, 0.0, 0.5], [0.0, 0.1, -0.1], 'Alpha Centauri D', "tab:green")
    
    # solve the problem
    mbp.initialize()
    mbp_sol, limits = mbp.solve(relative_to_com=True)
    
    # create the line objects for plots
    # NOTE: Can't pass empy arrays into 3d plot
    lines = [ax.plot(sol[0, 0:1], sol[1, 0:1], sol[2, 0:1], 
             color=mbp.bodies[i].color)[0] for i, sol in enumerate(mbp_sol)]
    dots = [ax.plot([sol[0, 1]], [sol[1, 1]], sol[2, 1],
            color=mbp.bodies[i].color, linestyle="", marker="o")[0] for i, sol in enumerate(mbp_sol)]
    
    ax.set_xlim3d([limits[0][0], limits[0][1]])
    ax.set_xlabel("x-coordinate", fontsize=14)
    ax.set_ylim3d([limits[1][0], limits[1][1]])
    ax.set_ylabel("y-coordinate", fontsize=14)
    ax.set_zlim3d([limits[2][0], limits[2][1]])
    ax.set_zlabel("z-coordinate", fontsize=14)
    ax.set_title("Visualization of orbits of stars in a two-body system\n", fontsize=14)
    # ax.legend(loc="upper left", fontsize=14)
    
    frames = sci.linspace(0, mbp.points - 1, num=(int(ANIMATION_LENGTH_SEC * 1000 / 50) - 1), dtype="int")[2:]
    ani = animation.FuncAnimation(fig, update_lines,
                                  frames=frames,
                                  fargs=(mbp_sol, lines, dots), repeat_delay=50, blit=False)
 
    plt.show()

