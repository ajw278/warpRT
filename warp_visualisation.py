import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# Load input data
i0_deg = 21.0
i0 = np.radians(i0_deg)
PA0 = np.radians(30)

# Rotate to warped orientation
def rotate(x, y, z, inc, PA):
    # Rotate by PA around z
    x1 = x*np.cos(PA) - y*np.sin(PA)
    y1 = x*np.sin(PA) + y*np.cos(PA)
    z1 = z
    # Rotate by inclination about x
    x2 = x1
    y2 = y1*np.cos(inc) - z1*np.sin(inc)
    z2 = y1*np.sin(inc) + z1*np.cos(inc)
    return x2, y2, z2

# Flatten the disc (for visualization)
def flatten_disc(x, y, z, i0, PA0):
    y1 = y*np.cos(-i0) - z*np.sin(-i0)
    z1 = y*np.sin(-i0) + z*np.cos(-i0)
    x1 = x
    x2 = x1*np.cos(-PA0) - y1*np.sin(-PA0)
    y2 = x1*np.sin(-PA0) + y1*np.cos(-PA0)
    z2 = z1
    return x2, y2, z2


# Warped inclination and PA profiles
radii, delta_i, delta_PA = np.loadtxt('mwc758_warpprofile.dat', unpack=True)


# High-resolution grid for polygons
N_r_poly = 100
N_phi_poly = 200
radii_poly = np.linspace(radii.min(), radii.max(), N_r_poly)
phi_poly = np.linspace(0, 2*np.pi, N_phi_poly)

# Lower-resolution grid for wireframe
N_r_line = 25
N_phi_line = 40
radii_line = np.linspace(radii.min(), radii.max(), N_r_line)
phi_line = np.linspace(0, 2*np.pi, N_phi_line)

# Observational residuals
xygrid = np.load('MWC 758_xygrid.npy')
dv_obs = np.load('MWC 758_dv_obs.npy')



R_grid_poly, phi_grid_poly = np.meshgrid(radii_poly, phi_poly, indexing='ij')
i_grid_poly = i0 + np.interp(R_grid_poly, radii, delta_i)
PA_grid_poly = PA0 + np.interp(R_grid_poly, radii, delta_PA)

x0_poly = R_grid_poly * np.cos(phi_grid_poly)
y0_poly = R_grid_poly * np.sin(phi_grid_poly)
z0_poly = np.zeros_like(x0_poly)

xw_poly, yw_poly, zw_poly = rotate(x0_poly, y0_poly, z0_poly, i_grid_poly, PA_grid_poly)
x_flat_poly, y_flat_poly, z_flat_poly = flatten_disc(xw_poly, yw_poly, zw_poly, i0, PA0)

R_grid_line, phi_grid_line = np.meshgrid(radii_line, phi_line, indexing='ij')
i_grid_line = i0 + np.interp(R_grid_line, radii, delta_i)
PA_grid_line = PA0 + np.interp(R_grid_line, radii, delta_PA)

x0_line = R_grid_line * np.cos(phi_grid_line)
y0_line = R_grid_line * np.sin(phi_grid_line)
z0_line = np.zeros_like(x0_line)

xw_line, yw_line, zw_line = rotate(x0_line, y0_line, z0_line, i_grid_line, PA_grid_line)
x_flat_line, y_flat_line, z_flat_line = flatten_disc(xw_line, yw_line, zw_line, i0, PA0)





# Build interpolator
x_axis = xygrid[0]
y_axis = xygrid[1]
interp = RegularGridInterpolator(
    (x_axis, y_axis),
    dv_obs.T,  # Transpose because numpy meshgrid indexing differs
    bounds_error=False,
    fill_value=np.nan
)


# Normalize for coloring
vmin, vmax = -0.3, 0.3
norm = plt.Normalize(vmin=vmin, vmax=vmax)
cmap = plt.cm.seismic

points_poly = np.column_stack((x_flat_poly.ravel(), y_flat_poly.ravel()))
dv_on_surface = interp(points_poly)
facecolors = cmap(norm(dv_on_surface))


# Plot
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')


# Triangular surface colored by dv_obs
#surf = ax.plot_trisurf( x_flat.ravel(),y_flat.ravel(),  z_flat.ravel(), triangles=None,  facecolors=facecolors,  linewidth=0.2, edgecolor='none',)
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

tri = mtri.Triangulation(x_flat_poly.ravel(), y_flat_poly.ravel())


radii_vertices = np.sqrt(x_flat_poly.ravel()**2 + y_flat_poly.ravel()**2)
r_min = np.amin(radii)  # You can change this threshold
r_max = 230.0  # You can change this threshold
tri_mask = np.all((radii_vertices[tri.triangles] >= r_min)&(radii_vertices[tri.triangles] <= r_max), axis=1)
valid_triangles = tri.triangles[tri_mask]


verts = [
    list(
        zip(
            x_flat_poly.ravel()[triangle],
            y_flat_poly.ravel()[triangle],
            z_flat_poly.ravel()[triangle]
        )
    )
    for triangle in valid_triangles
]

facecolors_valid = facecolors[valid_triangles[:,0]]


poly = Poly3DCollection(
    verts,
    facecolors=facecolors_valid,
    edgecolor="none",
    alpha=1.0
)


# Add to axes
ax.add_collection3d(poly)


# Add colorbar
mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
mappable.set_array([])
cbar = fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.1)
cbar.set_label(r"$\delta v_{\mathrm{obs}}$ [km/s]")



z_lift = 1e-3

radii_wireframe = np.sqrt(x_flat_line**2 + y_flat_line**2)
for i in range(len(x_flat_line)):
    if np.all(radii_wireframe[i, :] < r_min) or np.all(radii_wireframe[i, :] > r_max):
        continue  # skip this ring entirely
    ax.plot(
        x_flat_line[i, :],
        y_flat_line[i, :],
        z_flat_line[i, :]+ z_lift,
        color="k",
        lw=0.5,
        alpha=0.7,
        zorder=10
    )


for j in range(N_phi_line):
    r_line = radii_wireframe[:, j]
    mask = (r_line >= r_min)& (r_line <= r_max)

    # If no points are outside, skip
    if not np.any(mask):
        continue

    # If all points are outside, plot full line
    if np.all(mask):
        ax.plot(
            x_flat_line[:, j],
            y_flat_line[:, j],
            z_flat_line[:, j]+z_lift,
            color="k",
            lw=0.5, 
            zorder=10
        )
        continue

    # Otherwise, clip to the part outside r_min
    # Find first index where r >= r_min
    idx_start = np.argmax(mask)
    idx_end = len(r_line) - np.argmax(mask[::-1])


    # Plot only from idx_start onward
    ax.plot(
        x_flat_line[idx_start:idx_end, j],
        y_flat_line[idx_start:idx_end, j],
        z_flat_line[idx_start:idx_end, j]+z_lift,
        color="k",
        lw=0.5, 
        zorder=10
    )


"""
# Add colorbar
mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
mappable.set_array([])
cbar = fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.1)
cbar.set_label(r"$\delta v_{\mathrm{obs}}$ [km/s]")"""

# Limits and labels
ax.set_zlim([-30.,30.])
ax.set_xlim([-200.,200.])
ax.set_ylim([-200.,200.])
ax.set_xlabel("X [au]")
ax.set_ylabel("Y [au]")
ax.set_zlabel("Z [au]")
ax.view_init(elev=30, azim=45)
# Hide panes
ax.xaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
ax.yaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
ax.zaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Remove ticks and tick labels
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

# Remove grid
ax.grid(False)

# Remove axis labels
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_zlabel("")

ax.set_axis_off()


#plt.tight_layout()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D

# Input parameters
i0_deg = 21.0
i0 = np.radians(i0_deg)

PA0 = np.radians(30)   # base position angle

N_phi = 60

radii, delta_i, delta_PA = np.loadtxt('mwc758_warpprofile.dat', unpack=True)
phi = np.linspace(0, 2*np.pi, N_phi)

# 2D grid
R_grid, phi_grid = np.meshgrid(radii, phi, indexing='ij')

# Compute inclination and PA for each radius
i_grid = i0 + np.interp(R_grid, radii, delta_i)
PA_grid = PA0 + np.interp(R_grid, radii, delta_PA)

# Base positions in flat plane
x0 = R_grid * np.cos(phi_grid)
y0 = R_grid * np.sin(phi_grid)
z0 = np.zeros_like(x0)

# Rotate all points to their warped orientation
def rotate(x, y, z, inc, PA):
    # Rotate by PA about z
    x1 = x*np.cos(PA) - y*np.sin(PA)
    y1 = x*np.sin(PA) + y*np.cos(PA)
    z1 = z
    # Rotate by inc about x
    x2 = x1
    y2 = y1*np.cos(inc) - z1*np.sin(inc)
    z2 = y1*np.sin(inc) + z1*np.cos(inc)
    return x2, y2, z2

xw, yw, zw = rotate(x0, y0, z0, i_grid, PA_grid)

# Flatten the disc (optional)
def flatten_disc(x, y, z, i0, PA0):
    y1 = y*np.cos(-i0) - z*np.sin(-i0)
    z1 = y*np.sin(-i0) + z*np.cos(-i0)
    x1 = x
    x2 = x1*np.cos(-PA0) - y1*np.sin(-PA0)
    y2 = x1*np.sin(-PA0) + y1*np.cos(-PA0)
    z2 = z1
    return x2, y2, z2

x_flat, y_flat, z_flat = flatten_disc(xw, yw, zw, i0, PA0)

# Plot
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')

# Draw radial lines
for j in range(N_phi):
    ax.plot(x_flat[:, j], y_flat[:, j], z_flat[:, j], color='k', alpha=0.4, lw=0.8)

# Draw azimuthal rings
for i in range(len(x_flat)):
    ax.plot(x_flat[i, :], y_flat[i, :], z_flat[i, :], color='k', alpha=0.4, lw=0.8)

ax.plot_trisurf(
    x_flat.ravel(), y_flat.ravel(), z_flat.ravel(),
    color="lightblue", alpha=0.6, edgecolor="None", linewidth=0.2
)
ax.set_zlim([-20.,20.])
ax.set_xlim([-200.,200.])
ax.set_ylim([-200.,200.])
ax.set_xlabel("X [au]")
ax.set_ylabel("Y [au]")
ax.set_zlabel("Z [au]")
ax.view_init(elev=30, azim=45)
#ax.set_box_aspect([1, 1, 0.5])

plt.tight_layout()
plt.show()



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
ax.set_xlim([-200.,200.])
ax.set_ylim([-200.,200.])
ax.set_xlabel('X (au)')
ax.set_ylabel('Y (au)')
ax.set_zlabel('Z (au)')
ax.set_box_aspect([1, 1, 1])
ax.set_axis_off()

plt.tight_layout()
plt.show()
