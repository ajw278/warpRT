
import numpy as np
import matplotlib.pyplot as plt
from mpl_setup import *
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from radmc3dPy import analyze
from scipy.interpolate import RegularGridInterpolator
import os 

import matplotlib.patches as patches

R_MAX = 210.0
# Load the file
file_path = 'azimuthal_peakintensity_residuals_mwc758_13co.dat'

# Read all lines
with open(file_path, 'r') as f:
    lines = f.readlines()

radii = []
x_vals = []
y_vals = []
residuals = []

for line in lines:
    parts = line.strip().split()
    if len(parts) < 3:
        continue

    r = float(parts[0])
    azis_str = parts[1]
    temps_str = parts[2]

    azis_deg = np.array(list(map(float, azis_str.split(','))))
    temps = np.array(list(map(float, temps_str.split(','))))
    if len(azis_deg) != len(temps):
        continue

    azis_rad = np.radians(azis_deg)
    x = r * np.cos(azis_rad)
    y = r * np.sin(azis_rad)

    radii.extend([r] * len(azis_deg))
    x_vals.extend(x)
    y_vals.extend(y)
    residuals.extend(temps)

# Convert to arrays and AU
AU = 1.495978707e13
x_vals = np.array(x_vals) 
y_vals = np.array(y_vals) 
residuals = np.array(residuals)

R_MIN= np.amin(radii)
# Define grid
grid_size = 300
x_grid = np.linspace(-R_MAX, R_MAX, grid_size)
y_grid = np.linspace(-R_MAX, R_MAX, grid_size)
X, Y = np.meshgrid(x_grid, y_grid)

# Interpolate
Z_file = griddata(
    points=(x_vals, y_vals),
    values=residuals,
    xi=(X, Y),
    method='linear',
    fill_value=np.nan
)

R_grid = np.sqrt(X**2 + Y**2)
Z_file[(R_grid > R_MAX) | (R_grid < R_MIN)] = np.nan
print(R_grid)

# --- Load or compute CO temperature map and residuals ---
if not os.path.isfile('tmap_co.npy') or not os.path.isfile('xy_surf.npy') or not os.path.isfile('z_surf.npy'):
    dust = analyze.readData(ddens=True)
    dust.readDustTemp()

    r = dust.grid.x
    theta = dust.grid.y
    phi = dust.grid.z
    phi[-1] = 2.*np.pi
    phi[0] = 0.0

    rho_dust = dust.rhodust[:, :, :, 0]
    temp = dust.dusttemp
    rho_gas = rho_dust / 1e-2

    RR, TT, PP = np.meshgrid(r, theta, phi, indexing='ij')
    XX = RR * np.sin(TT) * np.cos(PP)
    YY = RR * np.sin(TT) * np.sin(PP)
    ZZ = RR * np.cos(TT)

    nx, ny, nz = 200, 200, 200
    x_cart = np.linspace(-r.max(), r.max(), nx)
    y_cart = np.linspace(-r.max(), r.max(), ny)
    z_cart = np.linspace(-r.max(), r.max(), nz)
    XXc, YYc, ZZc = np.meshgrid(x_cart, y_cart, z_cart, indexing='ij')

    Rc = np.sqrt(XXc**2 + YYc**2 + ZZc**2)
    Thetac = np.arccos(np.clip(ZZc / Rc, -1.0, 1.0))
    Phic = np.arctan2(YYc, XXc) % (2*np.pi)

    interp_rho = RegularGridInterpolator((r, theta, phi), rho_gas, bounds_error=False, fill_value=0.0)
    interp_temp = RegularGridInterpolator((r, theta, phi), temp, bounds_error=False, fill_value=np.nan)

    points = np.stack([Rc.ravel(), Thetac.ravel(), Phic.ravel()], axis=-1)
    rho_cart = interp_rho(points).reshape(nx, ny, nz)
    temp_cart = interp_temp(points).reshape(nx, ny, nz)

    dz = np.abs(z_cart[1] - z_cart[0])
    N_H = np.cumsum(rho_cart[:, :, ::-1] * dz / (2.3 * 1.67e-24), axis=2)
    N_CO = N_H * 1e-6

    thresh = 1e15
    mask = N_CO > thresh
    idx = np.argmax(mask, axis=2)
    print(idx)

    z_emit = z_cart[::-1][idx]
    temp_emit = np.take_along_axis(temp_cart[:, :, ::-1], idx[:, :, None], axis=2).squeeze()

    np.save('xy_surf.npy', np.array([x_cart, y_cart]))
    np.save('z_surf.npy', z_emit)
    np.save('tmap_co.npy', temp_emit)
    np.save('idsurf_map.npy', idx)
else:
    temp_emit = np.load('tmap_co.npy')
    x_cart, y_cart = np.load('xy_surf.npy')
    z_emit = np.load('z_surf.npy')
    idx= np.load('idsurf_map.npy')
   


iinc_x = np.absolute(x_cart)/AU<R_MAX
iinc_y = np.absolute(y_cart)/AU<R_MAX

XX_emit, YY_emit = np.meshgrid(x_cart[iinc_x], y_cart[iinc_y], indexing='ij')
RR_emit = np.sqrt(XX_emit**2 + YY_emit**2)
print(RR_emit.shape)
print(temp_emit.shape)
print(iinc_x, iinc_y)
temp_emit = temp_emit[np.where(iinc_x)[0]][:,np.where(iinc_y)[0]]
z_emit = z_emit[np.where(iinc_x)[0]][:,np.where(iinc_y)[0]]
print(temp_emit.shape)

rvals = RR_emit.ravel()
tvals = temp_emit.ravel()
x_cart = x_cart[iinc_x]

y_cart = y_cart[iinc_y]

nbins = 100
r_bins = np.logspace(np.log10(rvals.min()+1e-10), np.log10(rvals.max()), nbins+1)
r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
t_avg = np.zeros(nbins)

for i in range(nbins):
    mask = (rvals >= r_bins[i]) & (rvals < r_bins[i+1])
    t_avg[i] = np.nanmean(tvals[mask]) if np.any(mask) else np.nan

rmin = 30 * AU
rmax = 100 * AU
valid = (r_centers >= rmin) & (r_centers <= rmax) & (~np.isnan(t_avg))
r_fit = r_centers[valid]
t_fit = t_avg[valid]

def powerlaw(R, T0, q):
    return T0 * (R / r_fit[0])**(-q)

popt, _ = curve_fit(powerlaw, r_fit, t_fit, p0=[t_fit[0], 0.5])
T0_fit, q_fit = popt

R_emit_cm = np.sqrt(XX_emit**2 + YY_emit**2)
T_model = powerlaw(R_emit_cm, T0_fit, q_fit)
T_resid = temp_emit - T_model
T_resid[(R_emit_cm/AU > R_MAX) | (R_emit_cm/AU < R_MIN)] = np.nan

import matplotlib.gridspec as gridspec

# Create figure with wider layout to fit colorbar on the side
fig = plt.figure(figsize=(13, 5))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.15)

ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
cax = fig.add_subplot(gs[2])  # Colorbar axis



# Plot left panel
im0 = ax0.pcolormesh(X, Y, Z_file, cmap='PuOr', shading='auto', vmin=-13, vmax=13)
ax0.set_title('Temperature brightness')
"""ax0.set_xlabel('x [au]')
ax0.set_ylabel('y [au]')
ax0.set_xlim([-R_MAX, R_MAX])
ax0.set_ylim([-R_MAX, R_MAX])"""


from scipy.ndimage import gaussian_filter
# Beam FWHM in au
fwhm_arcsec = 0.15
distance_pc = 156
fwhm_au = fwhm_arcsec * distance_pc  # ~23.4 au

# Grid pixel size
dx = np.abs(x_cart[1] - x_cart[0]) / AU
dy = np.abs(y_cart[1] - y_cart[0]) / AU
pixel_size = 0.5 * (dx + dy)
sigma_pix = fwhm_au / (2.355 * pixel_size)

# Cylindrical radius grid (in au)
XX_emit, YY_emit = np.meshgrid(x_cart / AU, y_cart / AU, indexing='ij')
RR_emit = np.sqrt(XX_emit**2 + YY_emit**2)

# Create mask inside valid radius range
valid_mask = (RR_emit >= R_MIN) & (RR_emit <= R_MAX)

# Fill invalid regions with 0 and apply Gaussian filter
T_resid_filled = np.where(valid_mask, T_resid, 0.0)
T_resid_weight = valid_mask.astype(float)

# Apply Gaussian filter to both the filled map and the mask
T_resid_blur = gaussian_filter(T_resid_filled, sigma=sigma_pix)
weight_blur = gaussian_filter(T_resid_weight, sigma=sigma_pix)

# Avoid division by zero
with np.errstate(invalid='ignore', divide='ignore'):
    T_resid_smoothed = T_resid_blur / weight_blur
    T_resid_smoothed[weight_blur == 0] = np.nan  # restore nan outside original mask
    

T_resid_smoothed[RR_emit > R_MAX] = np.nan  # restore nan outside original mask


# Plot right panel
im1 = ax1.pcolormesh(x_cart / AU, y_cart / AU, T_resid_smoothed[::-1,:], cmap='PuOr', shading='auto', vmin=-13, vmax=13)
ax1.set_title('Model temperature')
"""ax1.set_xlabel('x [au]')
ax1.set_xlim([-R_MAX, R_MAX])
ax1.set_ylim([-R_MAX, R_MAX])"""

# Remove y-axis label and ticks from right-hand velocity map
	

for ax in [ax0, ax1]:
	#ax.set_aspect('equal')
	ax.set_xlabel("")
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_ylabel("")
	for spine in ax.spines.values():
		spine.set_visible(False)

# Choose round radius intervals
if R_MAX > 200:
	dr = 50
else:
	dr = 20

radii_au = np.arange(dr, R_MAX + 1, dr)  # e.g., [50, 100, 150, ...]

angles_deg = np.arange(0, 360, 45)

for ax in [ax0, ax1]:
	# Concentric circles
	for r in radii_au:
		circle = plt.Circle((0, 0), r, color='k', ls=':', lw=0.5, fill=False, zorder=20)
		ax.add_patch(circle)
		ax.text(r/np.sqrt(2), r/np.sqrt(2), f"{int(r)} au", fontsize=8, ha='left', va='bottom', rotation=0, color='k', zorder=21)

	
	# Radial lines
	for angle in angles_deg:
		angle_rad = np.deg2rad(angle)
		ax.plot([0, R_MAX * np.cos(angle_rad)], [0, R_MAX * np.sin(angle_rad)],
				color='k', ls=':', lw=0.5, zorder=20)

# Define the circle
circle_radius = 0.15 * 156 * 2  # in au
circle = patches.Circle((0, 0), radius=circle_radius, edgecolor='black', facecolor='gray', linestyle='--', linewidth=1.5)

# Add to both axes
ax0.add_patch(circle)
ax1.add_patch(patches.Circle((0, 0), radius=circle_radius, edgecolor='black', facecolor='gray', alpha=0.9, linestyle='--', linewidth=1.5))

# Add colorbar to the right of both plots
# Add shared colorbar
cbar = fig.colorbar(im1, cax=cax)
cbar.set_label('$\Delta T$ [K]')

plt.tight_layout()
plt.savefig('temperate_mwc758.png', bbox_inches='tight', format='png')
plt.show()

R_emit = np.sqrt(XX_emit**2+ YY_emit**2)

print(z_emit)
print(z_emit/R_emit)
# Plot
plt.figure()
plt.pcolormesh(x_cart/AU, y_cart/AU, z_emit/R_emit/AU, shading='auto', vmin=0.1, vmax=0.6)
#plt.xlabel('R [cm]')
#plt.ylabel('z_emission [cm]')
plt.title('CO Emission Height')
plt.colorbar(label='z_emission [cm]')
plt.show()

plt.figure()
plt.pcolormesh(x_cart, y_cart, temp_emit, shading='auto', vmin=5., vmax=120.0)
plt.xlabel('R [cm]')
plt.ylabel('T_emission [K]')
plt.title('Temperature at CO Emission Surface')
plt.colorbar(label='Temperature [K]')
plt.show()
