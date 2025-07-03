import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

from temp_extract import extract_dT

# ---------- Utility functions ----------

# Rotate to warped orientation
def rotate(x, y, z, inc, PA):
    x1 = x*np.cos(PA) - y*np.sin(PA)
    y1 = x*np.sin(PA) + y*np.cos(PA)
    z1 = z
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

# ---------- Load data ----------
i0_deg = 21.0
i0 = np.radians(i0_deg)
PA0 = np.radians(30)

# Warp profile
radii, delta_i, delta_PA = np.loadtxt('mwc758_warpprofile.dat', unpack=True)

# Temperature residuals
x_vals, y_vals, radii_pts, residuals = extract_dT('azimuthal_peakintensity_residuals_mwc758.dat')


# ---------- Grids ----------
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


# ---------- Warped surface coordinates ----------
# Polygons
R_grid_poly, phi_grid_poly = np.meshgrid(radii_poly, phi_poly, indexing='ij')
i_grid_poly = i0 + np.interp(R_grid_poly, radii, delta_i)
PA_grid_poly = PA0 + np.interp(R_grid_poly, radii, delta_PA)
x0_poly = R_grid_poly * np.cos(phi_grid_poly)
y0_poly = R_grid_poly * np.sin(phi_grid_poly)
z0_poly = np.zeros_like(x0_poly)
xw_poly, yw_poly, zw_poly = rotate(x0_poly, y0_poly, z0_poly, i_grid_poly, PA_grid_poly)
x_flat_poly, y_flat_poly, z_flat_poly = flatten_disc(xw_poly, yw_poly, zw_poly, i0, PA0)

# Wireframe
R_grid_line, phi_grid_line = np.meshgrid(radii_line, phi_line, indexing='ij')
i_grid_line = i0 + np.interp(R_grid_line, radii, delta_i)
PA_grid_line = PA0 + np.interp(R_grid_line, radii, delta_PA)
x0_line = R_grid_line * np.cos(phi_grid_line)
y0_line = R_grid_line * np.sin(phi_grid_line)
z0_line = np.zeros_like(x0_line)
xw_line, yw_line, zw_line = rotate(x0_line, y0_line, z0_line, i_grid_line, PA_grid_line)
x_flat_line, y_flat_line, z_flat_line = flatten_disc(xw_line, yw_line, zw_line, i0, PA0)


# ---------- Interpolate residuals ----------
interp = LinearNDInterpolator(
    np.column_stack((x_vals, y_vals)),
    residuals,
    fill_value=np.nan
)

temp_on_surface = interp(
    np.column_stack((x_flat_poly.ravel(), y_flat_poly.ravel()))
)


# ---------- Mask triangles ----------
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

tri = mtri.Triangulation(x_flat_poly.ravel(), y_flat_poly.ravel())

radii_vertices = np.sqrt(x_flat_poly.ravel()**2 + y_flat_poly.ravel()**2)
r_min = np.amin(radii)
r_max = 230.0

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

# ---------- Coloring ----------
vmin_T = -5
vmax_T = +20
norm_T = plt.Normalize(vmin=vmin_T, vmax=vmax_T)
cmap_T = plt.cm.magma
facecolors_T = cmap_T(norm_T(temp_on_surface))
facecolors_T_valid = facecolors_T[valid_triangles[:,0]]


# ---------- Plot ----------
fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111, projection='3d')

poly_T = Poly3DCollection(
    verts,
    facecolors=facecolors_T_valid,
    edgecolor="none",
    alpha=1.0
)
ax.add_collection3d(poly_T)

# Colorbar
mappable = plt.cm.ScalarMappable(cmap=cmap_T, norm=norm_T)
mappable.set_array([])
cbar = fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.1)
cbar.set_label(r"$\Delta T_b$ [K]")


# Wireframe
radii_wireframe = np.sqrt(x_flat_line**2 + y_flat_line**2)
z_lift = 1e-3

# Azimuthal rings
for i in range(len(x_flat_line)):
    r_ring = radii_wireframe[i,:]
    if np.all(r_ring < r_min) or np.all(r_ring > r_max):
        continue
    ax.plot(
        x_flat_line[i,:],
        y_flat_line[i,:],
        z_flat_line[i,:]+z_lift,
        color="w",
        lw=0.5,
        alpha=0.7,
        zorder=10
    )

# Radial lines
for j in range(N_phi_line):
    r_line = radii_wireframe[:,j]
    mask = (r_line >= r_min)&(r_line <= r_max)
    if not np.any(mask):
        continue
    idx_start = np.argmax(mask)
    idx_end = len(r_line) - np.argmax(mask[::-1])
    ax.plot(
        x_flat_line[idx_start:idx_end, j],
        y_flat_line[idx_start:idx_end, j],
        z_flat_line[idx_start:idx_end, j]+z_lift,
        color="w",
        lw=0.5,
        alpha=0.7,
        zorder=10
    )

# Limits and labels
ax.set_zlim([-30,30])
ax.set_xlim([-200,200])
ax.set_ylim([-200,200])
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_zlabel("")
ax.view_init(elev=30, azim=45)

# Hide panes
ax.xaxis.pane.set_edgecolor((1,1,1,0))
ax.yaxis.pane.set_edgecolor((1,1,1,0))
ax.zaxis.pane.set_edgecolor((1,1,1,0))
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

ax.grid(False)
ax.set_axis_off()


plt.tight_layout()
plt.show()
