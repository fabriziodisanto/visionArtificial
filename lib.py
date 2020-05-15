from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

# pip install matplotlib

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# y = [3, 5, 2, 6, 8, 4, 2, 1, 7, 9]
# z = [3, 6, 2, 6, 7, 3, 1, 9, 0, 7]
#
x, y, z = axes3d.get_test_data()

ax.scatter(x, y, z, c='b', marker='o')

# ax.plot_wireframe(x, y, z, rstride=3, cstride=10)

ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')

plt.show()
