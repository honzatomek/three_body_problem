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


class OneBody:
    def __init__(self, name, mass, position, velocity, color):
        self.name = name
        self.mass = mass
        self.position = sci.array(position, dtype="float64")
        self.velocity = sci.array(velocity, dtype="float64")
        self.color = color


class MultiBodyProblem:

    def __init__(self, periods=8, points=500):
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

        # print(dv_dt, dr_dt)
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

    def solve(self):
        multi_body_solution = sci.integrate.odeint(self.MultiBodyEquations,
                                                   self.init_params,
                                                   self.time_span)
                                                   #  args=(self.G))
        r_sol = []
        limits = []
        for i, body in enumerate(self.bodies):
            r_sol.append(sci.transpose(multi_body_solution[:, i*3:(i + 1) * 3]))
        
        return r_sol, float(multi_body_solution.min()), float(multi_body_solution.max())


def update_lines(num, dataLines, lines, dots):
    for line, data in zip(lines, dataLines):
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
    
    for dot, data in zip(dots, dataLines):
        # dot._offsets3d = [float(data[0, num]), float(data[1, num]), float(data[2, num])]
        dot.set_data(data[0:2, num])
        dot.set_3d_properties(data[2, num])

    return lines, dots


# Create figure
fig = plt.figure(figsize=(15, 15))
ax = p3.Axes3D(fig)

# solve the problem
mbp = MultiBodyProblem()
mbp.add_body(1.1, [-0.5, 0.0, 0.0], [0.01, 0.01, 0.0], 'Alpha Centauri A', "darkblue")
mbp.add_body(0.907, [0.5, 0.0, 0.0], [-0.05, 0.0, -0.1], 'Alpha Centauri B', "tab:red")
mbp.add_body(1.2, [0.0, 1.0, 0.0], [0.0, -0.1, 0.0], 'Alpha Centauri C', "tab:green")
mbp.initialize()
mbp_sol, minval, maxval = mbp.solve()

# create the line objects
# NOTE: Can't pass empy arrays into 3d plot
# lines = [ax.plot(sol[0:1, 0], sol[0:1, 1], sol[0:1, 2])[0] for sol in mbp_sol]
lines = [ax.plot(sol[0, 0:1], sol[1, 0:1], sol[2, 0:1], color=mbp.bodies[i].color)[0] for i, sol in enumerate(mbp_sol)]
# dots = [ax.scatter(sol[0, 1], sol[1, 1], sol[2, 1], color=mbp.bodies[i].color, marker="o", s=100) for i, sol in enumerate(mbp_sol)] 
dots = [ax.plot([float(sol[0, 1])], [float(sol[1, 1])], float(sol[2, 1]), color=mbp.bodies[i].color, linestyle="", marker="o")[0] for i, sol in enumerate(mbp_sol)]

# ax.plot(r2_sol[:,0],r2_sol[:,1],r2_sol[:,2],color="tab:red")
# Plot the final positions of the stars
# ax.scatter(r1_sol[-1,0],r1_sol[-1,1],r1_sol[-1,2],color="darkblue",marker="o",s=100,label="Alpha Centauri A")
# ax.scatter(r2_sol[-1,0],r2_sol[-1,1],r2_sol[-1,2],color="tab:red",marker="o",s=100,label="Alpha Centauri B")
ax.set_xlim3d([minval, maxval])
ax.set_xlabel("x-coordinate", fontsize=14)
ax.set_ylim3d([minval, maxval])
ax.set_ylabel("y-coordinate", fontsize=14)
ax.set_zlim3d([minval, maxval])
ax.set_zlabel("z-coordinate", fontsize=14)
ax.set_title("Visualization of orbits of stars in a two-body system\n", fontsize=14)
# ax.legend(loc="upper left", fontsize=14)

# print(r1_sol.size)
ani = animation.FuncAnimation(fig, update_lines, frames=sci.linspace(2, mbp.points - 1, mbp.points - 3, dtype="int"), fargs=(mbp_sol, lines, dots), interval=50, blit=False)
# ani = animation.FuncAnimation(fig, update_lines, frames=sci.linspace(2, mbp.points, dtype="int"), fargs=(mbp_sol, lines), interval=25, blit=False)

plt.show()
