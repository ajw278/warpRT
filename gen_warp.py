import numpy as np
from scipy.interpolate import interp1d
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Physical constants
au = 1.49598e13  # cm
pc = 3.08572e18  # cm
ms = 1.98892e33  # g
rs = 6.96e10     # cm
ts = 5.78e3      # K
ls = 3.8525e33   # erg/s

# Monte Carlo parameters
nphot = int(1e6)


# Model parameters
rho0 = 1e-17
r0 = 10 * au

nr, ntheta, nphi = 80, 600, 100
rin_au, rout_au = 20.0, 200.0
rin, rout = rin_au * 1.496e13, rout_au * 1.496e13  # cm
r = np.logspace(np.log10(rin), np.log10(rout), nr)
theta = np.linspace(-np.pi, np.pi, ntheta)
phi = np.linspace(0.0, 2 * np.pi, nphi)
RHO0  =  1e-14
WARPFILE = 'hd135344_warpprofile.txt'


# Star parameters
mstar, rstar, tstar = ms, rs, ts
pstar = np.array([0., 0., 0.])

# ------------------------------
# Warp Table
# ------------------------------


warp_data = np.loadtxt(WARPFILE)  # your warp table from earlier
r_warp, dinc, dpa = warp_data[:,0]*au, warp_data[:,1], warp_data[:,2]

f_inc = interp1d(r_warp, dinc, bounds_error=False, fill_value=(dinc[0], dinc[-1]))
f_pa  = interp1d(r_warp, dpa, bounds_error=False, fill_value=(dpa[0], dpa[-1]))

r_cm = warp_data[:, 0] * 1.496e13
dinc = warp_data[:, 1]
dpa = warp_data[:, 2]
f_dinc = interp1d(r_cm, dinc, kind='linear', fill_value='extrapolate')
f_dpa = interp1d(r_cm, dpa, kind='linear', fill_value='extrapolate')


def l_vector(i0, delta_i, delta_PA):
    l0 = l0_vector(i0)
    zhat = np.array([0, 0, 1])
    e_i = np.cross(zhat, l0)
    e_i /= np.linalg.norm(e_i)
    e_PA = np.cross(l0, e_i)
    return l0 + delta_i * e_i + delta_PA * np.sin(i0) * e_PA


def l0_vector(i0, PA0=0.0):
    return np.array([
        -np.sin(i0) * np.sin(PA0),
         np.sin(i0) * np.cos(PA0),
         np.cos(i0)
    ])


def rotation_from_z_to_l(lvec):
    lvec = lvec / np.linalg.norm(lvec)
    z = np.array([0, 0, 1])
    v = np.cross(z, lvec)
    s = np.linalg.norm(v)
    c = np.dot(z, lvec)
    if s == 0:
        return np.identity(3)  # no rotation needed
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    return R



def read_warp_profile(filename='warp_profile.txt'):
    data = np.loadtxt(filename)
    rwarp = data[:, 0] * au
    dincl = data[:, 1]
    dpa = data[:, 2]
    return interp1d(rwarp, dincl, bounds_error=False, fill_value=0.0), \
           interp1d(rwarp, dpa, bounds_error=False, fill_value=0.0)

def make_grid():
    xi = np.linspace(-sizex, sizex, nx+1)
    yi = np.linspace(-sizey, sizey, ny+1)
    zi = np.linspace(-sizez, sizez, nz+1)
    xc = 0.5 * (xi[:-1] + xi[1:])
    yc = 0.5 * (yi[:-1] + yi[1:])
    zc = 0.5 * (zi[:-1] + zi[1:])
    return xi, yi, zi, xc, yc, zc



# ------------------------------
# Density Model
# ------------------------------
def vertical_density(z, H):
    return  np.exp(-(z ** 2) / (2 * H ** 2))

def compute_density_warped(i0=np.deg2rad(50.)):
	rho = np.zeros((nr, ntheta, nphi), dtype=np.float64)
	rr, tt, pp = np.meshgrid(r, theta,phi, indexing='ij')

	H = 0.05 * r[0]*(r/r[0])**1.5
	H = H[:, np.newaxis, np.newaxis]

	rho0 = RHO0 * (r[:, np.newaxis, np.newaxis] / rin) ** -1.0


	x = rr * np.sin(tt) * np.cos(pp)
	y = rr * np.sin(tt) * np.sin(pp)
	z = rr * np.cos(tt)

	rho = np.zeros((nr, ntheta, nphi), dtype=np.float64)

	for i in range(nr):
		delta_i = f_dinc(r[i])
		delta_pa = f_dpa(r[i])
		print(delta_i, delta_pa)

		l_vec = l_vector(i0, delta_i, delta_pa)
		Rmat = rotation_from_z_to_l(l_vec)

		# Rotate x,y,z at this radius
		x_i = x[i]
		y_i = y[i]
		z_i = z[i]
		coords = np.stack([x_i, y_i, z_i], axis=-1)  # shape (ntheta, nphi, 3)
		coords_rot = coords @ Rmat.T  # rotate all vectors at once

		x_rot, y_rot, z_rot = coords_rot[..., 0], coords_rot[..., 1], coords_rot[..., 2]
		Rwarp = np.sqrt(x_rot**2 + y_rot**2)
		rho0 = RHO0 * (r[i] / r0) ** -1.0
		rho[i] = rho0*vertical_density(z_rot, H[i])

	return rho

# ------------------------------
# RADMC-3D File Writers
# ------------------------------
def write_amr_grid():
    with open("amr_grid.inp", "w") as f:
        f.write("0\n")  # regular format, not extended
        f.write("100\n")  # spherical grid
        f.write(f"{nr} {ntheta} {nphi}\n")

        # Write grid coordinates (cell edges!)
        r_edge = np.geomspace(rin, rout, nr + 1)
        theta_edge = np.linspace(0, np.pi, ntheta + 1)
        phi_edge = np.linspace(0, 2 * np.pi, nphi + 1)

        np.savetxt(f, r_edge)
        np.savetxt(f, theta_edge)
        np.savetxt(f, phi_edge)

def write_dust_density(rho):
    with open("dust_density.inp", "w") as f:
        f.write("1\n")
        f.write(f"{nr * ntheta * nphi}\n")
        np.savetxt(f, rho.ravel(order="F"))

def write_dustopac():
    with open("dustopac.inp", "w") as f:
        f.write("1\n1\nsilicate\n0\n1\n")
        f.write("dustkappa_silicate.inp\n\n")

def write_star():
    with open("stars.inp", "w") as f:
        f.write("2\n1\n1 0 0 0 0\n")
        f.write("1.989e33 6.96e10 5778\n0 0 0\n")

def write_radmc3d_inp():
    with open("radmc3d.inp", "w") as f:
        f.write("nphot = 1000000\n")
        f.write("scattering_mode_max = 1\n")
        f.write("istar_sphere = 1\n")


def write_wavelength_file(filename="wavelength_micron.inp", nlam=100, lam_min=0.1, lam_max=1e5):
    """Write wavelength_micron.inp with log-spaced grid for RADMC-3D."""
    lam = np.logspace(np.log10(lam_min), np.log10(lam_max), nlam)
    with open(filename, "w") as f:
        f.write(f"{nlam}\n")
        for val in lam:
            f.write(f"{val:.5g}\n")
    print(f"✅ Wrote {filename} with {nlam} wavelengths from {lam_min} to {lam_max} microns.")


# ------------------------------
# Visualization: Density Slice
# ------------------------------
def plot_density_slice(rho):
	"""
	Plot a full polar vertical slice from the warped 3D density cube.
	Uses actual values at φ = 0 (right side) and φ = π (left side).
	"""
	print("📊 Plotting full polar density slice from φ=0 and φ=π...")

	print(rho.shape)

	# Grab φ = 0 and φ = π slices
	phi_index_0 = 0
	phi_index_pi = rho.shape[2] // 2
	rho_phi0 = rho[:, :, phi_index_0]
	rho_phipi = rho[:, :, phi_index_pi]

	# Build R and Z grids
	theta_grid = np.tile(theta, (nr, 1))
	r_grid = np.tile(r[:, None], (1, ntheta))
	z_grid = r_grid * np.cos(theta_grid) # - np.pi / 2)

	# Convert to AU
	r_au = r_grid / 1.496e13
	z_au = z_grid / 1.496e13

	# Concatenate φ=π and φ=0 into one full polar slice
	r_full = np.concatenate((-r_au[::-1], r_au), axis=0)
	z_full = np.concatenate((z_au[::-1], z_au), axis=0)
	rho_full = np.concatenate((rho_phipi[::-1], rho_phi0), axis=0)

	# Plot
	import matplotlib.pyplot as plt
	import matplotlib.colors as colors

	plt.figure(figsize=(10, 6))
	plt.pcolormesh(r_full, z_full/r_full, rho_full, shading='auto', norm=colors.LogNorm(vmin=1e-16, vmax=1e-14))
	#plt.scatter(r_full.flatten(), z_full.flatten()/r_full.flatten(), s=1, c='r')
	plt.xlabel('Radius [AU] (φ = π on left, φ = 0 on right)')
	plt.ylabel('Height [AU]')
	plt.title('Warped Disc Density Slice (Actual φ = 0 and φ = π)')
	#plt.ylim([-30.0, 30.0])
	plt.colorbar(label='Dust Density [g/cm³]')
	plt.tight_layout()
	plt.savefig("density_slice_fullpolar.png", dpi=150)
	plt.show()

# ------------------------------
# Main Setup Function
# ------------------------------
def setup_model():
    os.makedirs("radmc3d_model", exist_ok=True)
    os.chdir("radmc3d_model")

    print("Writing grid...")
    write_amr_grid()

    print("Computing density cube with warp...")
    rho = compute_density_warped()

    print("📈 Plotting density slice...")
    plot_density_slice(rho)
    
    write_dust_density(rho)

    print("Writing other files...")
    write_dustopac()
    write_star()
    write_radmc3d_inp()
    write_wavelength_file()

    print("\n✅ Setup complete.")
    print("Now run:")
    print("  radmc3d mctherm")
    print("  radmc3d image lambda 1.25 incl 45 sca")

if __name__ == "__main__":
    setup_model()

