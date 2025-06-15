import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Input parameters
i0_deg = 21.0
i0 = np.radians(i0_deg)

# Functions for orientation and rotation
def l0_vector(i0, PA0=0.0):
    return np.array([
        -np.sin(i0) * np.sin(PA0),
         np.sin(i0) * np.cos(PA0),
         np.cos(i0)
    ])

def l_vector(i0, delta_i, delta_PA):
    l0 = l0_vector(i0)
    zhat = np.array([0, 0, 1])
    e_i = np.cross(zhat, l0)
    e_i /= np.linalg.norm(e_i)
    e_PA = np.cross(l0, e_i)
    return l0 + delta_i * e_i + delta_PA * np.sin(i0) * e_PA

def rotation_from_z_to_l(lvec):
    lvec = lvec / np.linalg.norm(lvec)
    z = np.array([0, 0, 1])
    v = np.cross(z, lvec)
    s = np.linalg.norm(v)
    c = np.dot(z, lvec)
    if s == 0:
        return np.identity(3)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    return R

# Load your data (replace this with your actual file path)
data = np.loadtxt('mwc758_warpprofile.dat')
radius, inclination_deg, pa_deg = data.T
inclination_deg *= 180.0/np.pi
pa_deg *= 180.0/np.pi

# Interpolation
fine_radius = np.linspace(radius.min(), radius.max(), 300)
incl_interp = interp1d(radius, inclination_deg, kind='cubic')
pa_interp = interp1d(radius, pa_deg, kind='cubic')
fine_inclination = incl_interp(fine_radius)
fine_pa = pa_interp(fine_radius)

# Reference rotation to flatten the disc
l0_anti = l_vector(-i0, 0.0, 0.0)
Rx_minus_i0 = rotation_from_z_to_l(l0_anti)

# Create a flat ring in the xy plane
ntheta = 200
theta = np.linspace(0, 2 * np.pi, ntheta)
ring_xy = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=-1)  # (ntheta, 3)

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for r, inc_deg, pa_deg_val in zip(fine_radius, fine_inclination, fine_pa):
    delta_i = np.radians(inc_deg)
    delta_pa = np.radians(pa_deg_val)
    lvec = l_vector(i0, delta_i, delta_pa)
    Rwarp = rotation_from_z_to_l(lvec)

    coords = ring_xy * r
    coords_warped = coords @ Rwarp.T
    coords_faceon = coords_warped @ Rx_minus_i0.T  # Should flatten back into x-y

    x_rot, y_rot, z_rot = coords_faceon.T
    ax.plot(x_rot, y_rot, z_rot, color=plt.cm.viridis((r - radius.min()) / (radius.max() - radius.min())), alpha=0.25)

# View should now be face-on
ax.view_init(elev=45, azim=45)
ax.set_zlim([-20.,20.])
ax.set_xlabel('X (au)')
ax.set_ylabel('Y (au)')
ax.set_zlabel('Z (au)')
ax.set_box_aspect([1, 1, 1])
plt.tight_layout()
plt.show()
