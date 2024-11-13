import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np

def onClick(event):
	global pause
	pause ^= True

# SET HERE WHAT TO READ
path = 'new_data/test_github/'
filename = path + "output_N50.csv"
N = 50
fps = 5  # frames per second 
frames=1000

# read data from output file
data = pd.read_csv(filename)
x_min = data["x"].min()
x_max = data["x"].max()

y_min = np.min(data['MaZe']) - 0.15
y_max = np.max(data['MaZe']) + 0.15

# initializing a figure
fig = plt.figure(figsize=(18.5, 9.9))
 
# marking the x-axis and y-axis
ax = fig.add_subplot(1,1,1)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_title('Evolution electrostatic potential', fontsize=24)
ax.set_xlabel('x (a.u.)', fontsize = 20)
ax.set_ylabel('Potential (a.u.)', fontsize = 20)

# initializing a line variable
line, = ax.plot([], [], lw = 1, marker='o', color='r', label='MaZe', linestyle='--')
 
ax.legend(fontsize=16)

def init():
	line.set_data([], [])
	return line,
 
def animate(i):
	line.set_data(data.loc[data["iter"].astype(int) == i]["x"].to_numpy(), data.loc[data["iter"].astype(int) == i]["MaZe"].to_numpy())
	return line, 

fig.canvas.mpl_connect('button_press_event', onClick)    
anim = animation.FuncAnimation(fig, animate, init_func = init, frames = frames, blit=True) 

#writergif = animation.PillowWriter(fps=fps)
writerffm = animation.FFMpegWriter(fps=fps)

# save animation
anim.save(path + f'animation_N' + str(N) + '_fps' + str(fps) + '.mp4', writer = writerffm)
print('\nAnimation ready: saved as animation_N' + str(N) + '_fps' + str(fps) + '_dt1.mp4 \n')
 