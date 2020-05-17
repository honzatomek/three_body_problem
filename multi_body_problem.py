#!/usr/bin/python3

# <---------------------------------------------------------------------------- general imports --->
import os
import sys
import datetime
import configparser # Import configparser to read and save configurations
# <---------------------------------------------------------------------------- numeric imports --->
import math
# Import scipy
import scipy as sci 
import scipy.integrate
# Import matplotlib and associated modules for 3D and animations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
# from matplotlib import animation
import matplotlib.animation as animation


# <--------------------------------------------------------------------------- global variables --->
ANIMATION_LENGTH_SEC = 10
# CONFIG_INI = os.path.splitext(__file__)[0] + '.ini'
CONFIG_INI = os.path.join(os.path.dirname(__file__), 'save')
print(__file__)

# <------------------------------------------------------------------------------------ classes --->
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
        print('[\033[01;32m+\033[0m] initialising multibody problem solver\n    periods: {0},integration points: {1}'.format(periods, points))
        self.periods = periods
        self.points = points

        # define universal gravitation constant
        self.g = 5.67408e-11  # n-m2/kg2
        # reference quantities
        self.m_nd = 0.989e+30  # kg #mass of the sun
        self.r_nd = 4.326e+12  # m #distance between stars in alpha centauri
        self.v_nd = 29999  # m/s #relative velocity of earth around the sun
        self.t_nd = 78.91 * 365 * 24 * 3600 * 0.51  # s #orbital period of alpha centauri
        # net constants
        self.k0= self.g * self.t_nd * self.m_nd / (self.r_nd ** 2 * self.v_nd)
        self.k1 = self.v_nd * self.t_nd / self.r_nd

        self.bodies = []

        self.com = None

        self.init_params = None
        self.time_span = None

    def add_body(self, mass, position, velocity, name=None, color=None):
        if name is None:
            name = 'star {0}'.format(len(self.bodies) + 1)
        if color is None:
            color = (random(), random(), random())
        self.bodies.append(OneBody(name, mass, position, velocity, color))
        print('[\033[01;32m+\033[0m] added celestial body:\n{0}'.format(str(self.bodies[-1])))
        self.init_com()

    def init_com(self):
        r_numerator = 0
        v_numerator = 0
        denominator = 0
        for body in self.bodies:
            r_numerator += body.mass * body.position
            v_numerator += body.mass * body.velocity
            denominator += body.mass

        self.com = OneBody('com', denominator, r_numerator / denominator, v_numerator / denominator, "tab:yellow")

    #a function defining the equations of motion
    def multibodyequations(self, w, t):  # , g):
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
            dr_dt.append(self.k1 * v[i])
            for j in range(num):
                if i != j:
                    dv_dt[i] += self.k0 * self.bodies[j].mass * (r[j] - r[i]) / (sci.linalg.norm(r[j] - r[i]) ** 3)

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
        print('[\033[01;32m+\033[0m] running solver')
        multi_body_solution = sci.integrate.odeint(self.multibodyequations,
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

        print('[\033[01;32m+\033[0m] output data relative to COG of system: {0}'.format(str(relative_to_com)))
        self.solution = r_sol
        self.limits = limits
        # return r_sol, limits

    def load(self, filepath):
        print('[\033[01;32m+\033[0m] loading saved sessions')
        if os.path.isfile(filepath) and filepath.endswith('.ini'):
            files = [filepath]
            filepath = os.path.dirname(filepath)
        elif os.path.isdir(filepath):
            files = [f for f in os.listdir(filepath) if f.endswith('.ini')]
        else:
            print('[\033[01;31m-\033[0m] missing directory with saved files')
            sys.exit(1)

        for i, f in enumerate(files):   
            c = configparser.ConfigParser()
            c.read_file(open(os.path.join(filepath, f)))
            print('\t{0} - {1} ({2}), {3} bodies'.format(i, f, c['DEFAULT']['description'], c['DEFAULT']['bodies']))
            for j in range(int(c['DEFAULT']['bodies'])):
                print('\t\t{0:<20} m = {1: 6}\tx = [{2: 6}, {3: 6}, {4: 6}]\tv = [{5: 6}, {6: 6}, {7: 6}]\tcolor = {8:<20}'.format(c[str(j)]['name'],
                                                                                                                                   float(c[str(j)]['mass']),
                                                                                                                                   float(c[str(j)]['x']),
                                                                                                                                   float(c[str(j)]['y']),
                                                                                                                                   float(c[str(j)]['z']),
                                                                                                                                   float(c[str(j)]['vx']),
                                                                                                                                   float(c[str(j)]['vy']),
                                                                                                                                   float(c[str(j)]['vz']),
                                                                                                                                   c[str(j)]['color']))
        select = input('[\033[01;34m?\033[0m] Select session number to load [0]: ')
        if select == '':
            select = '0'
        if select in [str(i) for i in range(len(files))]:
            select = int(select)
        else:
            sys.exit('[\033[01;31m-\033[0m] no session selected, exiting...')
        
        c = configparser.ConfigParser()
        c.read_file(open(os.path.join(filepath, files[select])))
        self.periods = int(c['DEFAULT']['periods to solve'])
        self.points = int(c['DEFAULT']['integration points'])
        for j in range(int(c['DEFAULT']['bodies'])):
            self.add_body(float(c[str(j)]['mass']), [float(c[str(j)]['x']), float(c[str(j)]['y']), float(c[str(j)]['z'])], [float(c[str(j)]['vx']), float(c[str(j)]['vy']), float(c[str(j)]['vz'])], c[str(j)]['name'], c[str(j)]['color'])

    def save(self, filepath):
        defname = '{0}'.format(timestamp('fullname'))
        session = input('[\033[01;34m?\033[0m] session name [{0}]: '.format(defname))
        if session == '':
            session = defname

        description = input('[\033[01;34m?\033[0m] description: ')
        
        filename = os.path.join(filepath, session.replace(' ', '_').replace(':', '') + '.ini')
        print('[\033[01;32m+\033[0m] saving session to: {0}'.format(filename))

        config = configparser.ConfigParser(interpolation=None)
        config['DEFAULT']['name'] = session
        config['DEFAULT']['description'] = description
        config['DEFAULT']['date'] = timestamp('datename')
        config['DEFAULT']['time'] = timestamp('time')
        config['DEFAULT']['bodies'] = str(len(self.bodies))
        config['DEFAULT']['periods to solve'] = str(self.periods)
        config['DEFAULT']['integration points'] = str(self.points)
        
        for i, b in enumerate(self.bodies):
            config.add_section(str(i))
            config[str(i)]['name'] = b.name
            config[str(i)]['mass'] = '{0:e}'.format(b.mass)
            for j, x in enumerate(['x', 'y', 'z']):
                config[str(i)][x] = '{0:e}'.format(b.position[j]) 
            for j, x in enumerate(['vx', 'vy', 'vz']):
                config[str(i)][x] = '{0:e}'.format(b.velocity[j])
            config[str(i)]['color'] = str(b.color)

        # print(session)
        # config.read_dict(session)
        
        if not os.path.isdir(os.path.dirname(filename)):
            os.mkdir(os.path.dirname(filename))
        with open(filename, 'w') as configfile:
            config.write(configfile)
        print('[\033[01;32m+\033[0m] session saved')
    
    def plot(self):
        print('[\033[01;32m+\033[0m] plotting solution')

        # Create figure
        fig = plt.figure(figsize=(9, 9))
        ax = p3.Axes3D(fig)

        # create the line objects for plots
        # NOTE: Can't pass empy arrays into 3d plot
        lines = [ax.plot(sol[0, 0:1], sol[1, 0:1], sol[2, 0:1], 
                 color=self.bodies[i].color)[0] for i, sol in enumerate(self.solution)]
        dots = [ax.plot([sol[0, 1]], [sol[1, 1]], sol[2, 1],
                color=self.bodies[i].color, linestyle="", marker="o")[0] for i, sol in enumerate(self.solution)]
        
        def update_lines(num, dataLines, lines, dots):
            for line, data in zip(lines, dataLines):
                line.set_data(data[0:2, :num])
                line.set_3d_properties(data[2, :num])
            
            for dot, data in zip(dots, dataLines):
                # dot._offsets3d = [float(data[0, num]), float(data[1, num]), float(data[2, num])]
                dot.set_data(data[0:2, num])
                dot.set_3d_properties(data[2, num])

            return lines, dots

        ax.set_xlim3d([self.limits[0][0], self.limits[0][1]])
        ax.set_xlabel("x-coordinate", fontsize=14)
        ax.set_ylim3d([self.limits[1][0], self.limits[1][1]])
        ax.set_ylabel("y-coordinate", fontsize=14)
        ax.set_zlim3d([self.limits[2][0], self.limits[2][1]])
        ax.set_zlabel("z-coordinate", fontsize=14)
        ax.set_title("Visualization of orbits of stars in a two-body system\n", fontsize=14)
        # ax.legend(loc="upper left", fontsize=14)
        
        frames = sci.linspace(0, self.points - 1, num=(int(ANIMATION_LENGTH_SEC * 1000 / 50) - 1), dtype="int")[2:]
        ani = animation.FuncAnimation(fig, update_lines,
                                      frames=frames,
                                      fargs=(self.solution, lines, dots), repeat_delay=25, blit=False)
 
        plt.show()


# <----------------------------------------------------------------------------- help functions --->
def query_yes_no(question: str='Continue?'):
    prompt = '[\033[01;34m?\033[0m] ' + question + ' [Y/n]: '
    answer = input(prompt)
    if answer.lower() in ['y', 'yes']:
        return True
    elif answer.lower() in ['n', 'no']:
        return False
    else:
        return True


def timestamp(out='fullname'):
    """
    Returns timestamp, possible outputs: fullname [full, date, datename ,time]
    fullname: 2020-Feb-01 07:28:15
    full:     20200201_072815
    datename: 2020-Feb-01
    date:     2020-02-01
    time:     07:28:15
    """
    if out == 'fullname':
        return '{0:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())
    elif out == 'full':
        return '{0:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    elif out == 'datename':
        return '{0:%Y-%b-%d}'.format(datetime.datetime.now())
    elif out == 'date':
        return '{0:%Y-%m-%d}'.format(datetime.datetime.now())
    elif out == 'time':
        return '{0:%H:%M:%S}'.format(datetime.datetime.now())
    else:
        return '{0:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())


# <------------------------------------------------------------------------------ main function --->
def multi_body_problem():
    print('[\033[01;32m+\033[0m] started script to plot Multi-Body Problem using ODE')
    
    # initialize class
    mbp = MultiBodyProblem(20, 1500)
    
    # set up the celestial bodies
    if query_yes_no('do you wand to load saved session?'):
        mbp.load(CONFIG_INI)
    else:
        mbp.add_body(1.1, [-0.5, 0.0, 0.0], [0.01, 0.01, 0.0], 'Alpha Centauri A', "tab:blue")
        mbp.add_body(0.907, [0.5 , 0.0, 0.0], [-0.05, 0.0, -0.1], 'Alpha Centauri B', "tab:red")
        mbp.add_body(0.585, [0.5, 1.0, 0.0], [0.5, -0.05, 0.0], 'Alpha Centauri C', "tab:purple")
        # mbp.add_body(0.985, [0.0, 1.0, 0.0], [0.0, -0.005, 0.0], 'Alpha Centauri C', "tab:purple")
    
    # solve the problem
    mbp.initialize()
    # mbp_sol, limits = mbp.solve(relative_to_com=True)
    mbp.solve(relative_to_com=True)
    mbp.plot()
   
    if query_yes_no('save the configuration?'):
        mbp.save(CONFIG_INI)


# <---------------------------------------------------------------------------- main entrypoint --->
if __name__ == '__main__':
    multi_body_problem()
