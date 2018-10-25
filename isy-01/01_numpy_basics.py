import numpy as np

# A1. Numpy and Linear Algebra

results = []

a = np.zeros((10))
a[4] = 1
print('a', a)

b = np.arange(10, 49)
print('b', b)

print('c', b[::-1])

d = np.arange(0, 16).reshape(4, 4)
print('d', d)

e = np.random.random((4, 4))
min = e.min()
max = e.max()
e -= min
e *= 1 / (max - min)
print('e', e)

f = np.arange(4 * 3).reshape((4, 3)) @ np.arange(3 * 2).reshape((3, 2))
print('f', f)

g = np.arange(0, 21)
g = np.where((8 <= g) & (g <= 16), -g, g)
print('g', g)

print('h', a.sum())

i = np.arange(5 * 5).reshape(5, 5)
evenMask = np.fromfunction(lambda x, y: (y % 2) == 0, (1, 5))[0]
oddMask = np.invert(evenMask)
print('i', i[evenMask, :])
print('i', i[oddMask, :])

jM = np.arange(4 * 3).reshape((4, 3))
jV = np.arange(3)
print('j', jM * jV)


k = np.random.random((10, 2))
r = np.sqrt(np.square(k[:,0])+np.square(k[:,1]))
angle = np.arctan(k[:,0]/k[:,1])
p = np.column_stack((r,angle))
print('k', p)

def vLen(v):
    return v.shape[0]


def dot(v, scalar):
    res = []
    i = 0
    while i < vLen(v):
        res.append(v[i] * scalar)
        i += 1
    return np.array(res)


v1 = np.array([1, 2, 3, 4, 5])
v2 = np.array([-1, 9, 5, 3, 1])
for v in [v1, v2]:
    print('l', vLen(v))
    print('l', dot(v, 10))

M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 2, 2]])
v0 = np.array([1, 1, 0])
v1 = np.array([-1, 2, 5])
print('m', (v0.transpose() * v1) * M * v0)
