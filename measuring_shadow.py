import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

end_positions = pd.read_csv('end_positions_small_step.csv', index_col=0)
end_positions.columns = ['x', 'y', 'z']

# Drop runaways
end_positions = end_positions[end_positions['x'] <= 10]
end_positions = end_positions[end_positions['x'] >= 0]

# Drop points in the top kite -- can't project them
end_positions = end_positions[np.logical_or(np.logical_and(end_positions['x'] <= 5,
                                            end_positions['y'] <= -2.12467 * end_positions['x'] + 12.4432),
                                            np.logical_and(end_positions['x'] > 5,
                                            end_positions['y'] <= 2.12467 * end_positions['x'] - 8.8035))]

# Project points onto central lines, y=0.47066x and y=-0.47066x + 4.7066
line1_vector = np.array([1, 0.47066])
line2_vector = np.array([1, -0.47066])

norm_sq_1 = np.dot(line1_vector, line1_vector)
norm_sq_2 = np.dot(line2_vector, line2_vector)

projections = []

for index, row in end_positions.iterrows():
    pos_vector = np.array([row['x'], row['y']])
    if row['x'] <= 5:  # Project onto the first line
        projection = np.dot(pos_vector, line1_vector) * line1_vector / norm_sq_1
        projections.append(list(projection))
    else:  # Project onto the second line then shift to second half of the room
        proj_1 = np.dot(pos_vector, line2_vector) * line2_vector / norm_sq_2
        projection = proj_1 + np.array([1.8135, 3.8536])
        projections.append(list(projection))

projections_array = np.array(projections)
end_positions['line_x'] = projections_array[:, 0]  # Add projected x values to dataframe
y = gaussian_kde(projections_array[:, 0])(projections_array[:, 0])  # Get density values
y_new = []
for i in range(len(y)):  # Normalize by room width
    if projections_array[i, 0] <= 5:
        d = 0.193392 * projections_array[i, 0]  # Constant obtained by solving triangles at x for upper and lower walls
    else:
        d = -0.193392 * projections_array[i, 0] + 1.93392
    y_new.append(y[i] / d)


plt.figure(figsize=(10, 6))
plt.style.use('dark_background')
plt.grid(True, which='both', color='gray', alpha=0.4)
plt.scatter(projections_array[:, 0], y_new, color='y')
plt.yscale('log')
plt.xlabel('Room X-projection'); plt.ylabel('Log of normalized photon density')
plt.title('Relative photon density distribution of Tokarsky room')
plt.annotate('Light source', (0, 18.5), (1, 18.5), arrowprops={'arrowstyle': '->', 'color': 'white'})
plt.annotate('"Unilluminable" spot', (10, 2.6), (7, 2.6), arrowprops={'arrowstyle': '->', 'color': 'white'})
plt.savefig('normalized_density_small_step.png')
plt.show()

# Distribution after normalization should be continuous uniform--displacement confirms light or dark regions
