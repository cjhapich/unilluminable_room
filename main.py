# This code will establish the brightness of various points in the 4-sided Tokarski room

### GAME PLAN ###

# * Establish the boundaries of the room as a boolean 'crossed wall' function
#   ~ Establish walls as line functions
#     ~ NOTE!! Bottom walls go through the room, so need condition to check if x is in first or second half
#   ~ Function takes particle x position, calculates all limiting y positions, and calls reflect if
#     particle is outside
# * Create photon object with attributes location, velocity, and in_bounds & functions move, reflect,
#   wall cross, and floor cross
#   X Location and velocity initialized with the object, then updated with move and reflect functions
#  !! Move function updates location by stepping in velocity (the smaller the steps the better because
#     this is only producing the end scatter points--no accuracy/frame number tradeoff)
#   X Reflect function is activated when in_bounds is false and takes velocity vector and wall normal vector
#     and returns new velocity vector
#     ~     v2 = v1 - 2*(v1 dot n)*n      where n is normalized wall vector pointing inward
#   X Wall cross works as described above
#   X Floor cross checks z-dimension sign. If negative, removes particle and returns velocity to
#     scatter list
# * Simulate N photons with equal downward velocities and random planar velocities from random heights
#   above the origin until the floor has been crossed, at which point return the location
# * Plot the boundaries of the room as lines and the photon locations as scatter points

# Room is defined as a quadrilateral with a 120 degree vertex at the top and 10 degree vertices on the
# left and right

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

np.random.seed(1)


class Photon:

    # Normal vectors to the walls: 1 is top left, 2 is bottom left, 3 is top right, 4 is bottom right
    NORM_1 = np.array([0.5, -0.86602, 0])
    NORM_2 = np.array([-0.34202, 0.93969, 0])
    NORM_3 = np.array([-0.5, -0.86602, 0])
    NORM_4 = np.array([0.34202, 0.93969, 0])

    def __init__(self, pos=None, vel=None):  # Position and velocity passed as lists
        if vel is None:
            vel = [0, 0, 0]
        self.position = np.array(pos)
        self.velocity = np.array(vel)
        self.in_bounds = True
        self.traveling = True
        self.buffer = False  # Some particles bounce too far out and reflect indefinitely--this stops them

    def reflect(self, wall_norm):
        self.velocity = self.velocity - (2 * np.dot(self.velocity, wall_norm) * wall_norm)

    def check_wall_cross(self):
        if self.position[0] == 0 and self.position[1] == 0:
            return
        elif self.position[0] <= 5:
            if self.position[1] >= self.position[0] * 0.57735:  # Check wall 1
                self.reflect(Photon.NORM_1)
                self.buffer = True
            elif self.position[1] <= self.position[0] * 0.36397:  # Check wall 2
                self.reflect(Photon.NORM_2)
                self.buffer = True
            else:
                return
        else:
            if self.position[1] >= (self.position[0] * -0.57735) + 5.7735:  # Check wall 3
                self.reflect(Photon.NORM_3)
                self.buffer = True
            elif self.position[1] <= (self.position[0] * -0.36397) + 3.6397:  # Check wall 4
                self.reflect(Photon.NORM_4)
                self.buffer = True
            else:
                return

    def check_floor_cross(self):
        if self.position[2] <= 0:
            self.traveling = False

    def move(self, step):
        if not self.buffer:
            self.check_wall_cross()
        else:
            self.buffer = False
        self.check_floor_cross()
        self.position = self.position + (self.velocity * step)


def normalize(array):  # Normalize the x and y coordinates of a vector
    for i in range(len(array)):
        x = array[i][0]
        y = array[i][1]
        scale = np.sqrt(x ** 2 + y ** 2)
        array[i][0] = x / scale
        array[i][1] = y / scale
    return array


''' Simulate photon end points '''

photon_num = 100000
start_pos = []
start_vel = []
for i in range(photon_num):  # Create position and velocity vectors
    start_pos += [[0, 0, 10 * np.random.random()]]
    # Start velocities defined within the bounds of the walls
    start_vel += [[1, 0.21338 * np.random.random() + 0.36397, -0.05]]
start_vel = normalize(start_vel)

# Initialize photon objects
photons = []
for i in range(photon_num):
    photons.append(Photon(start_pos[i], start_vel[i]))

# Simulate photon motion
for i in range(len(photons)):
    while photons[i].traveling:
        photons[i].move(0.001)

# Get final positions
end_positions = []
for i in range(len(photons)):
    end_positions += [photons[i].position]
end_positions = np.array(end_positions)
positions_df = pd.DataFrame(end_positions)
positions_df.to_csv('end_positions_small_step.csv')  # Written out for quicker analysis

# Plot the shadow
x1 = np.linspace(0, 5, 10)
x2 = np.linspace(5, 10, 10)
plt.style.use('dark_background')
plt.figure(figsize=(20, 6))
plt.plot(x1, x1 * 0.57735, c='silver'); plt.plot(x1, x1 * 0.36397, c='silver')  # Plot the walls
plt.plot(x2, x2 * -0.57735 + 5.7736, c='silver'); plt.plot(x2, x2 * -0.36397 + 3.6396, c='silver')
plt.scatter(end_positions[:, 0], end_positions[:, 1], s=0.05)  # Shadow points
plt.title('Photon Distribution in a Four-Sided Tokarsky Unilluminable Room')
plt.xlim([-0.1, 10.1])
plt.ylim([-0.1, 3])
plt.xticks([]); plt.yticks([])
#plt.savefig('Tokarsky_room100k_.png')
plt.show()

''' Track photon paths 

photon_num = 1000
start_pos = []
start_vel = []
for i in range(photon_num):  # Create position and velocity vectors
    start_pos += [[0, 0, 10 * np.random.random()]]
    # Start velocities defined within the bounds of the walls
    start_vel += [[1, 0.21338 * np.random.random() + 0.36397, -0.05]]
start_vel = normalize(start_vel)

# Initialize photon objects
photons = []
for i in range(photon_num):
    photons.append(Photon(start_pos[i], start_vel[i]))

plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 6))
ax = fig.add_subplot(111, xlim=(-.1, 1.1), ylim=(-.1, 1.1))
x1 = np.linspace(0, 5, 10)
x2 = np.linspace(5, 10, 10)
plt.plot(x1, x1 * 0.57735, c='silver'); plt.plot(x1, x1 * 0.36397, c='silver')  # Plot the walls
plt.plot(x2, x2 * -0.57735 + 5.7736, c='silver'); plt.plot(x2, x2 * -0.36397 + 3.6396, c='silver')
plt.xlim([-0.1, 10.1])
plt.ylim([-0.1, 3])
plt.xticks([]); plt.yticks([])
evol, = ax.plot([], [], 'yo', ms=1)
xdata, ydata = [], []
x, y = [], []
for i in range(photon_num):
    x.append(photons[i].position[0])
    y.append(photons[i].position[1])
xdata.append(x)
ydata.append(y)


def init():
    evol.set_data([], [])
    return evol


def animate(frame):
    x, y = [], []
    for i in range(photon_num):
        for j in range(10):  # Number of steps to take before recording position for animation
            photons[i].move(0.001)
        x.append(photons[i].position[0])
        y.append(photons[i].position[1])
    evol.set_data(x, y)
    return evol


ani = animation.FuncAnimation(fig, animate, frames=30000, interval=10, init_func=init)
ani.save('particles_long_small_step.mp4')
'''

''' Simulate ray tracing 

start_vel = [2, .85, 0]
p1 = Photon([0, 0, 1], start_vel)

plt.style.use('dark_background')
fig = plt.figure(figsize=(10, 3))
axis = plt.axes(xlim=(-0.1, 10.1), ylim=(-0.1, 3))
line, = axis.plot([], [])
xdata, ydata = [p1.position[0]], [p1.position[1]]


def init():
    line.set_data([], [])
    return line,


def animate(frame):
    p1.move(0.01)
    xdata.append(p1.position[0])
    ydata.append(p1.position[1])
    line.set_data(xdata, ydata)
    line.set_color('y')
    #line.set_linewidth(0.2)
    return line,


x1 = np.linspace(0, 5, 10)
x2 = np.linspace(5, 10, 10)
# Plot the walls, offset by 0.01 for better visualization
plt.plot(x1, x1 * 0.57735 + 0.01, c='silver', linewidth=2); plt.plot(x1, x1 * 0.36397 - 0.01, c='silver', linewidth=2)
plt.plot(x2, x2 * -0.57735 + 5.7836, c='silver', linewidth=2)
plt.plot(x2, x2 * -0.36397 + 3.6296, c='silver', linewidth=2)
plt.xticks([]); plt.yticks([])
anim = animation.FuncAnimation(fig, animate, init_func=init, interval=4, frames=7000)

anim.save('ray_trace.mp4')
'''