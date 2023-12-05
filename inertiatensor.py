import numpy as np
import matplotlib.pyplot as plt

def tens_update(nb, coord, tens, restchannels):
    nb = nb[:-restchannels]
    nb_ = np.append(nb[1:], nb[0])
    values = np.multiply(nb, nb_)

    v1 = values[0] + values[3]
    v2 = values[1] + values[4]
    v3 = values[2] + values[5]

    v = [v1, v2, v3]

    if v[1] > v[2] and v[1] > v[0]:
        tens[(coord)] = np.array([[0.5, 0], [0, -0.5]])

    if v[0] > v[1] and v[0] > v[2]:
        tens[(coord)] = np.array([[0, -0.5], [-0.5, 0]])

    if v[2] > v[0] and v[2] > v[1]:
        tens[(coord)] = np.array([[0, 0.5], [0.5, 0]])

short= np.sin(np.pi/6)
long = np.cos(np.pi/6)
x_0 = [0, 0, 0, 0, 0, 0]
y_0 = [0, 0, 0, 0, 0, 0]
x = [1, short, -short, -1, -short, short]
y = [0, long, long, 0, -long, -long]
weights =  [0.5, 0.1, 0.7, 0.9, 0.2, 0.7]

def in_tensor (x, y):
    tens = []
    for i in np.arange(len(x)):
        tens.append(weights[i] * np.array([[y[i]**2, -x[i]*y[i]], [-x[i]*y[i], x[i]**2]]))
    return tens

träg = in_tensor(x, y)
b = sum(träg)
a = np.dot(b, np.array([1/np.sqrt(2),1/np.sqrt(2)]))
ev, a2 = np.linalg.eig(b)
print(ev, a2)
if ev[0] >= ev[1]:
    a2 = a2[::-1]

print(a2)

x_0 = [0, 0, 0, 0, 0, 0, 0, 0]
y_0 = [0, 0, 0, 0, 0, 0, 0, 0]
x = [1, -short, short, -1, -short, short, a2[0][0], a2[1][0]]
y = [0, long, long, 0, -long, -long, a2[0][1], a2[1][1]]
fig, ax = plt.subplots()
plt.quiver(x_0, y_0, x, y, scale = 4, color =["b", 'b', 'b', 'b', 'b', 'b', 'g', 'y'])

print("green = Massemittelpunkt", "yellow = resulting guidance vector")

print("vector", a2[1])
b = np.outer(a2[1], a2[1]) - 0.5 * np.diag(np.ones(2))
print("tensor", b)

plt.show()

