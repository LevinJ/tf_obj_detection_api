import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig1 = plt.figure()

# Fixing random state for reproducibility
np.random.seed(19680801)

data = np.random.rand(2, 25)
l, = plt.plot([], [], 'r-')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('x')
plt.title('test')

def update_line(num, data, line):
    line.set_data(data[..., :num])
    return line,

line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
                                   interval=50, blit=True)
line_ani.save('../data/lines.mp4', fps=5, dpi=300)

# plt.show()

# To save the animation, use the command: line_ani.save('lines.mp4')

print("done")