#!/usr/bin/python3

# <---------------------------------------------------------------------------- general imports --->
import os
import sys
# <--------------------------------------------------------------------------------- my imports --->
from threebodyproblem.other import *
from threebodyproblem.multibody import *

# <--------------------------------------------------------------------------- global variables --->
CONFIG_INI = os.path.join(os.path.dirname(__file__), 'save')


# <------------------------------------------------------------------------------ main function --->
def multi_body_problem():
    fprint('[+] started script to plot Multi-Body Problem using ODE')
    
    # initialize class
    mbp = MultiBodyProblem(20, 2000)
    
    # set up the celestial bodies
    s = query_choice('Select session input', ['Script', 'Load', 'Manual', 'Exit'])
    if s == 'load':
        mbp.load(CONFIG_INI)
    elif s == 'manual':
        i = 0
        while True:
            s = fprint('[?] Input data for body {0} [name; mass; x; y; z; vx; vy; vz] (input nothing to stop): '.format(i + 1), question=True)
            if s == '':
                break
            
            #try:
            vals = s.split(';')
            print(vals)
            mbp.add_body(float(vals[1]), [float(v) for v in vals[2:5]], [float(v) for v in vals[5:8]],vals[0]) 
            i += 1
            # except Exception as e:
            #     fprint('[-] Wrong input, check the values and try agian\n    {0}'.format(s))

    elif s == 'script':
        mbp.add_body(1.1, [-0.5, 0.0, 0.0], [0.01, 0.01, 0.0], 'Alpha Centauri A', "tab:blue")
        mbp.add_body(0.907, [0.5 , 0.0, 0.0], [-0.05, 0.0, -0.1], 'Alpha Centauri B', "tab:red")
        mbp.add_body(0.585, [0.5, 1.0, 0.0], [0.5, -0.05, 0.0], 'Alpha Centauri C', "tab:purple")
        # mbp.add_body(0.985, [0.0, 1.0, 0.0], [0.0, -0.005, 0.0], 'Alpha Centauri C', "tab:purple")
    else:
        sys.exit(fprint('[-] script exiting...', returnstr=True))
    
    # solve the problem
    mbp.initialize()
    mbp.solve(relative_to_com=True)
    mbp.plot()
   
    if query_yes_no('save the configuration?'):
        mbp.save(CONFIG_INI)


# <---------------------------------------------------------------------------- main entrypoint --->
if __name__ == '__main__':
    multi_body_problem()
