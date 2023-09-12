import starMovement1
import matplotlib.pyplot as plt

azh = starMovement1.starMovement(21, 52, 2023, 9, 13, [7, 46, 41.375], [27, 58, 12.03], 2)
azh2 = starMovement1.starMovement(21, 52, 2023, 9, 13, [7, 46, 41.375], [27, 58, 12.03], 2)

fig = plt.figure()
fig = starMovement1.skyplot(fig, azh)
fig = starMovement1.heightAzimuth(fig, azh2)
plt.show()