import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

#matplotlib.use('Agg')

plot_dir = sys.argv[1]

def load_states(filename):
    data = np.loadtxt(filename, delimiter=',')
    num_time_steps = data.shape[0]
    assert data.shape[1] == 5
    x = data[:,0]
    y = data[:,1]
    measured_bearing = data[:,4]
    return (x, y, measured_bearing)

def draw_bearing(bearing):
    radius = 5
    xs = [0.0, radius * np.cos(bearing)]
    ys = [0.0, radius * np.sin(bearing)]
    plt.plot(xs, ys, color='red')

def plot(x, y, measured_bearing):
    plt.scatter(x, y)
    for bearing in measured_bearing:
        draw_bearing(bearing)
    plt.xlim((-10, 10))
    plt.ylim((-10, 10))
    
plt.figure()
(x, y, measured_bearing) = load_states(plot_dir + '/ground_truth.csv')
plot(x, y, measured_bearing)
plt.savefig(plot_dir + "/ground_truth.png")
plt.close("all")

plt.figure(figsize=(32,32))
for i in range(100):
    print(i)
    plt.subplot(10, 10, i+1)
    (x, y, measured_bearing) = load_states(plot_dir + '/' + str(i) + '.csv')
    plot(x, y, measured_bearing)
plt.savefig(plot_dir + "/inferred.png")
plt.tight_layout()
plt.close("all")
