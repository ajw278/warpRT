import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from mpl_setup import *

# Physical constants
au = 1.49598e13  # cm
pc = 3.08572e18  # cm
ms = 1.98892e33  # g
rs = 6.96e10     # cm
ts = 5.78e3      # K
ls = 3.8525e33   # erg/s

# Monte Carlo parameters
nphot = int(5e5)

# Grid parameters
nx, ny, nz = 512, 512, 512
sizex, sizey, sizez = 120*au, 120*au, 120*au

# Model parameters
RHO0 = 2e-15
r0 = 10 * au

H_R0 = 0.05
flang =0.2

rin = 10 * au
rout = 260*au

# Star parameters
mstar, rstar, tstar = 1.4*ms,2.*rs, 7600.0
pstar = np.array([0., 0., 0.])

THETA_OPEN = 0.0

nr, ntheta, nphi = 200,200,200
r_edges = np.geomspace(rin, rout, nr+1)
theta_edges = np.linspace(THETA_OPEN,  np.pi - THETA_OPEN, ntheta+1)
phi_edges = np.linspace(0.0, 2 * np.pi, nphi+1)

r = (r_edges[:-1]+r_edges[1:])/2.
theta = (theta_edges[:-1]+theta_edges[1:])/2.
phi = (phi_edges[:-1]+phi_edges[1:])/2.

WARPFILE = 'mwc758_warpprofile.txt' #'hd135344_warpprofile.txt'

warp_data = np.loadtxt(WARPFILE)  # your warp table from earlier
r_warp, dinc, dpa = warp_data[:,0]*au, warp_data[:,1], warp_data[:,2]


#f_inc = interp1d(r_warp, dinc, bounds_error=False, fill_value=(dinc[0], dinc[-1]))
#f_pa  = interp1d(r_warp, dpa, bounds_error=False, fill_value=(dpa[0], dpa[-1]))

setup_number = 6
if setup_number==1:
	# Prepend (0.0, 0.0) to ensure warp goes to zero at origin
	r_ext = np.insert(r_warp, 0, 30.0*au)
	r_ext = np.insert(r_ext, 0, 0.0)
	dinc_ext = np.insert(dinc, 0, 0.00)
	dinc_ext = np.insert(dinc_ext, 0, 0.00)
	dpa_ext = np.insert(dpa, 0, 0.0)
	dpa_ext = np.insert(dpa_ext, 0, -0.05)
elif setup_number ==2:
	# Prepend (0.0, 0.0) to ensure warp goes to zero at origin
	r_ext = np.insert(r_warp, 0, 30.0*au)
	r_ext = np.insert(r_ext, 0, 0.0)
	dinc_ext = np.insert(dinc, 0, dinc[0])
	dinc_ext = np.insert(dinc_ext, 0, dinc[0])
	dpa_ext = np.insert(dpa, 0, dpa[0])
	dpa_ext = np.insert(dpa_ext, 0, dpa[0])
elif setup_number ==3:
	rin = 2 * au
	nr, ntheta, nphi = 400, 400, 400
	r_edges = np.geomspace(rin, rout, nr+1)
	theta_edges = np.linspace(THETA_OPEN,  np.pi - THETA_OPEN, ntheta+1)
	phi_edges = np.linspace(0.0, 2 * np.pi, nphi+1)

	r = (r_edges[:-1]+r_edges[1:])/2.
	theta = (theta_edges[:-1]+theta_edges[1:])/2.
	phi = (phi_edges[:-1]+phi_edges[1:])/2.

	# Prepend (0.0, 0.0) to ensure warp goes to zero at origin
	r_ext = np.insert(r_warp, 0, 30.0*au)
	r_ext = np.insert(r_ext, 0, 0.0)
	dinc_ext = np.insert(dinc, 0, dinc[0]*1.5)
	dinc_ext = np.insert(dinc_ext, 0, dinc[0]*2.0)
	dpa_ext = np.insert(dpa, 0, 0.0)
	dpa_ext = np.insert(dpa_ext, 0, -0.05)
     
elif setup_number ==4:
	rin = 10 * au
	flang =0.01
	nr, ntheta, nphi = 400, 400, 400
	r_edges = np.geomspace(rin, rout, nr+1)
	theta_edges = np.linspace(THETA_OPEN,  np.pi - THETA_OPEN, ntheta+1)
	phi_edges = np.linspace(0.0, 2 * np.pi, nphi+1)

	r = (r_edges[:-1]+r_edges[1:])/2.
	theta = (theta_edges[:-1]+theta_edges[1:])/2.
	phi = (phi_edges[:-1]+phi_edges[1:])/2.

	# Prepend (0.0, 0.0) to ensure warp goes to zero at origin
	r_ext = np.insert(r_warp, 0, 30.0*au)
	r_ext = np.insert(r_ext, 0, 0.0)
	dinc_ext = np.insert(dinc, 0, dinc[0])
	dinc_ext = np.insert(dinc_ext, 0, dinc[0])
	dpa_ext = np.insert(dpa, 0, 0.02)
	dpa_ext = np.insert(dpa_ext, 0, 0.0)


elif setup_number ==5:
	rin = 5 * au
	flang =0.05

	nr, ntheta, nphi = 400, 400, 400
	nr, ntheta, nphi = 100, 100, 100
	r_edges = np.geomspace(rin, rout, nr+1)
	theta_edges = np.linspace(THETA_OPEN,  np.pi - THETA_OPEN, ntheta+1)
	phi_edges = np.linspace(0.0, 2 * np.pi, nphi+1)

	r = (r_edges[:-1]+r_edges[1:])/2.
	theta = (theta_edges[:-1]+theta_edges[1:])/2.
	phi = (phi_edges[:-1]+phi_edges[1:])/2.

	# Prepend (0.0, 0.0) to ensure warp goes to zero at origin
    
	r_ext = np.insert(r_warp, 0, 30.0*au)
	r_ext = np.insert(r_ext, 0, rin)
		
	inc_ = 'random'
	pa_ = 'random'

	dinc1 = 0.1+0.08*np.random.uniform()
	dinc2= 0.07+0.1*np.random.uniform()
	dpa1 = 0.1+ 0.1*np.random.uniform()
	dpa2=0.05+0.1*np.random.uniform()
	flang = np.random.uniform()*0.04
	print(flang, dinc1, dinc2, dpa1, dpa2)
	flang= 0.03
	dinc1= 0.16573385299223117 *0.3
	dinc2= 0.11086120515157749 *0.3
	dpa1 =0.14124519749643605
	dpa2 = 0.14703113604509488
      
	flang= 0.2
	dinc1= 0.16573385299223117 *0.18
	dinc2= 0.11086120515157749 *0.32
	dpa1 =0.14124519749643605
	dpa2 = 0.05
      
	  
	flang= 0.3
	dinc1= 0.16573385299223117 *0.18
	dinc2= 0.11086120515157749 *0.32
	dpa1 =0.14124519749643605
	dpa2 = 0.05

	dinc_ext = np.insert(dinc, 0, dinc1)
	dinc_ext = np.insert(dinc_ext, 0, dinc2)
	dpa_ext = np.insert(dpa, 0, dpa1)
	dpa_ext = np.insert(dpa_ext, 0, dpa2)

    
	if inc_=='flat':
		dinc_ext = np.insert(dinc_ext, 0, dinc[0])
	elif inc_=='decrease':
		dinc_ext = np.insert(dinc, 0, 0.05)
		dinc_ext = np.insert(dinc_ext, 0, -0.1)
	elif inc_=='increase':
		dinc_ext = np.insert(dinc, 0, 1.3*dinc[0])
		dinc_ext = np.insert(dinc_ext, 0, 2.2*dinc[0])
            
	
	if pa_=='flat':
		dpa_ext = np.insert(dpa, 0, dpa[0])
		dpa_ext = np.insert(dpa_ext, 0, dpa[0])
	elif pa_=='decrease':
		dpa_ext = np.insert(dpa, 0, 0.1)
		dpa_ext = np.insert(dpa_ext, 0, -0.1)
	elif pa_=='increase':
		dpa_ext = np.insert(dpa, 0, 0.1)
		dpa_ext = np.insert(dpa_ext, 0, 0.2)



elif setup_number ==6:

	H_R0 = 0.05
	rin = 5 * au
	flang =0.03

	nr, ntheta, nphi = 400, 400, 400
	nr, ntheta, nphi = 200, 360, 200
	r_edges = np.geomspace(rin, rout, nr+1)
	theta_edges = np.linspace(THETA_OPEN,  np.pi - THETA_OPEN, ntheta+1)
	phi_edges = np.linspace(0.0, 2 * np.pi, nphi+1)

	r = (r_edges[:-1]+r_edges[1:])/2.
	theta = (theta_edges[:-1]+theta_edges[1:])/2.
	phi = (phi_edges[:-1]+phi_edges[1:])/2.

	# Prepend (0.0, 0.0) to ensure warp goes to zero at origin
    
	r_ext = np.insert(r_warp, 0, 30.0*au)
	r_ext = np.insert(r_ext, 0, rin)

	flang= 0.3
	dinc1= 0.16573385299223117 *0.18
	dinc2= 0.11086120515157749 *0.32
	dpa1 =0.14124519749643605
	dpa2 = 0.05
	dpa1 =0.00
	dpa2 = -0.2

	dinc_ext = np.insert(dinc, 0, dinc1)
	dinc_ext = np.insert(dinc_ext, 0, dinc2)
	dpa_ext = np.insert(dpa, 0, dpa1)
	dpa_ext = np.insert(dpa_ext, 0, dpa2)


	nphot = int(1e8)


			
     


# Create cubic spline interpolators (with extrapolation)
f_inc = CubicSpline(r_ext, dinc_ext, extrapolate=True)
f_pa  = CubicSpline(r_ext, dpa_ext, extrapolate=True)

from scipy.interpolate import CubicSpline


# Optional: plot to verify behavior
r_test = np.linspace(0, 280, 1000)*au
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(r_ext/au, dinc_ext,c='r')
plt.plot(r_warp/au, dinc, 'k.', label='Original')
plt.plot(r_test/au, f_inc(r_test), label='CubicSpline')
plt.axhline(0, color='gray', ls='--', lw=0.5)
plt.title('Delta Inclination')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(r_warp/au, dpa, 'k.', label='Original')
plt.plot(r_test/au, f_pa(r_test), label='CubicSpline')
plt.axhline(0, color='gray', ls='--', lw=0.5)
plt.title('Delta PA')
plt.legend()

plt.tight_layout()
plt.show()


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



def cartesian_grid():
	x = np.linspace(-sizex, sizex, nx)
	y = np.linspace(-sizey, sizey, ny)
	z = np.linspace(-sizez, sizez, nz)
	return np.meshgrid(x, y, z, indexing='ij')


from scipy.interpolate import RegularGridInterpolator

def interpolate_to_cartesian(rho_sph, r, theta, phi, x_grid, y_grid, z_grid):
	# Create interpolator
	interp = RegularGridInterpolator((r, theta, phi), rho_sph, bounds_error=False, fill_value=0)

	# Convert Cartesian grid to spherical coordinates
	rr = np.sqrt(x_grid**2 + y_grid**2 + z_grid**2)
	tt = np.arccos(z_grid / (rr + 1e-30))  # avoid division by zero
	pp = np.arctan2(y_grid, x_grid) % (2 * np.pi)

	# Stack points for interpolation
	interp_points = np.stack([rr, tt, pp], axis=-1)
	rho_cart = interp(interp_points)

	return rho_cart

# ------------------------------
# Density Model
# ------------------------------
def vertical_density(z, H):
    return  np.exp(-(z ** 2) / (2 * H ** 2))

def compute_density_warped(i0=np.deg2rad(21.),  M_star=1.4 * 1.98847e33,G = 6.67430e-8):
	rho = np.zeros((nr, ntheta, nphi), dtype=np.float64)
	vxyz = np.zeros((nr, ntheta, nphi, 3), dtype=np.float64)
	rr, tt, pp = np.meshgrid(r, theta,phi, indexing='ij')

	H = H_R0 * r0*(r/r0)**(1.+flang)
	H = H[:, np.newaxis, np.newaxis]

	rho0 = RHO0 * (r[:, np.newaxis, np.newaxis] / r0) ** -1.0


	x = rr * np.sin(tt) * np.cos(pp)
	y = rr * np.sin(tt) * np.sin(pp)
	z = rr * np.cos(tt)

	rho = np.zeros((nr, ntheta, nphi), dtype=np.float64)


	l0_antivect=  l_vector(-i0, 0.0, 0.0)
	
	Rx_minus_i0 = rotation_from_z_to_l(l0_antivect)

	for i in range(nr):
		
		delta_i = f_inc(r[i])
		delta_pa = f_pa(r[i])

		# Rotate x,y,z at this radius
		x_i = x[i]
		y_i = y[i]
		z_i = z[i]
		coords = np.stack([x_i, y_i, z_i], axis=-1)  # shape (ntheta, nphi, 3)
			

		l_vec = l_vector(i0, delta_i, delta_pa)
		Rwarp = rotation_from_z_to_l(l_vec)

		# Rotate x, y, z at this radius with warp
		coords = np.stack([x[i], y[i], z[i]], axis=-1)  # shape (ntheta, nphi, 3)
		coords_warped = coords @ Rwarp.T
		coords_faceon = coords_warped @ Rx_minus_i0.T

		x_rot, y_rot, z_rot = coords_faceon[..., 0], coords_faceon[..., 1], coords_faceon[..., 2]
		R_cyl = np.sqrt(x_rot**2 + y_rot**2)


		#x_rot, y_rot, z_rot = coords_rot[..., 0], coords_rot[..., 1], coords_rot[..., 2]
		
		rho0 = RHO0 * (r[i] / r0) ** -1.0
		rho[i] = rho0*vertical_density(z_rot, H[i])


		# Compute Keplerian azimuthal velocity
		vk = np.sqrt(G * M_star / np.maximum(R_cyl, 1e-5))  # avoid division by zero
				
		# Unit vector in phi direction (tangential)
		vphi_unit = np.stack([-y_rot / R_cyl, x_rot / R_cyl, np.zeros_like(R_cyl)], axis=-1)
		v_local = vk[..., np.newaxis] * vphi_unit  # (ntheta, nphi, 3)

		# Apply inverse warp (i.e., reverse the rotations)
		v_global = v_local @ Rx_minus_i0 @ Rwarp  # note: right to left multiplication
		vxyz[i]=  v_global

	return rho, vxyz

def write_amr_grid(xi, yi, zi, fname='amr_grid.inp'):
	with open(fname, 'w') as f:
		f.write("1\n")             # format
		f.write("0\n")             # regular grid
		f.write("0\n")             # cartesian
		f.write("0\n")             # no gridinfo
		f.write("1 1 1\n")         # include x, y, z
		f.write(f"{len(xi)-1} {len(yi)-1} {len(zi)-1}\n")
		for arr in (xi, yi, zi):
			for v in np.ravel(arr):
				f.write(f"{float(v):13.6e}\n")

def write_co_number_density(rho_sph, abundance=1e-5):
	mu = 2.3  # mean molecular weight
	mH = 1.6737e-24  # hydrogen mass in grams

	nCO = rho_sph * (abundance*100.0  / (mu * mH))# gas-to-dust ratio

	with open('numberdens_co.inp', 'w') as f:
		f.write("1\n")
		f.write(f"{nCO.size}\n")
		f.write("1\n")  # number of species
		nCO.ravel(order='F').tofile(f, sep='\n', format='%13.6e')
		f.write('\n')

def write_density(rhod):
	with open('dust_density.inp', 'w') as f:
		f.write('1\n')
		f.write(f"{nx * ny * nz}\n1\n")
		rhod.ravel(order='F').tofile(f, sep='\n', format="%13.6e")
		f.write('\n')

def write_wavelength_grid():
	lam1, lam2, lam3, lam4 = 0.1, 7.0, 25., 1e4
	n12, n23, n34 = 20, 100, 30
	lam = np.concatenate([
		np.logspace(np.log10(lam1), np.log10(lam2), n12, endpoint=False),
		np.logspace(np.log10(lam2), np.log10(lam3), n23, endpoint=False),
		np.logspace(np.log10(lam3), np.log10(lam4), n34, endpoint=True)
	])
	with open('wavelength_micron.inp', 'w') as f:
		f.write(f"{len(lam)}\n")
		f.writelines(f"{l:13.6e}\n" for l in lam)
	return lam

def write_stars(lam):
	with open('stars.inp', 'w') as f:
		f.write('2\n')
		f.write(f"1 {len(lam)}\n\n")
		f.write(f"{rstar:13.6e} {mstar:13.6e} {pstar[0]:13.6e} {pstar[1]:13.6e} {pstar[2]:13.6e}\n\n")
		f.writelines(f"{l:13.6e}\n" for l in lam)
		f.write(f"\n{-tstar:13.6e}\n")

def write_opacity_control():
	with open('dustopac.inp', 'w') as f:
		f.write('2\n1\n============================================================================\n')
		f.write('10\n0\nsilicate\n----------------------------------------------------------------------------\n')

def write_line_input():
	with open('lines.inp', 'w') as f:
		f.write('2\n1\n')
		f.write('co   leiden   0   0   0')

def write_radmc3d_inp():
	with open('radmc3d.inp', 'w') as f:
		f.write(f"nphot = {nphot}\niranfreqmode = 1\n")
		#Optical depth 5 for speedup
		f.write("mc_scat_maxtauabs = 5.d0\n")
		f.write("tgas_eq_tdust=1")

def plot_phi_slices(rhod, xc, yc, zc):
	from scipy.interpolate import RegularGridInterpolator
	phi_vals = [0.0, np.pi]
	fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
	interp = RegularGridInterpolator((xc, yc, zc), rhod, bounds_error=False, fill_value=np.nan)
	r = np.linspace(5 * au, 250 * au, 256)
	z = np.linspace(-30 * au, 30 * au, 256)
	R, Z = np.meshgrid(r, z, indexing='ij')
	for i, phi in enumerate(phi_vals):
		x = R * np.cos(phi)
		y = R * np.sin(phi)
		points = np.column_stack([x.ravel(), y.ravel(), Z.ravel()])
		rho_slice = interp(points).reshape(r.shape[0], z.shape[0])
		axes[i].imshow(np.log10(rho_slice + 1e-25), origin='lower', aspect='auto',
						extent=[-30, 30, r[0]/au, r[-1]/au], cmap='inferno')
		axes[i].set_title(f"phi = {phi:.1f} rad")
		axes[i].set_xlabel("z [au]")
	axes[0].set_ylabel("R [au]")
	plt.tight_layout()
	plt.savefig("density_phi_slices.png")
	plt.close()

def plot_density_slice(x, z, rho_cart):
	ny = rho_cart.shape[1] // 2
	fig, ax = plt.subplots(figsize=(6,5))
	im = ax.pcolormesh(x[:,0,0]/au, z[0,0,:]/au, np.log10(rho_cart[:, ny, :].T+1e-30), vmin=-18, vmax=np.log10(RHO0), shading='auto')
	ax.set_xlabel('x [au]')
	ax.set_ylabel('z [au]')
	plt.colorbar(im, ax=ax, label='log10(density)')
	plt.tight_layout()
	plt.show()

def compute_cell_walls(xc):
	dx = np.diff(xc)
	xw = np.zeros(len(xc) + 1)
	xw[1:-1] = 0.5 * (xc[:-1] + xc[1:])
	xw[0] = xc[0] - dx[0]/2
	xw[-1] = xc[-1] + dx[-1]/2
	return xw


def write_amr_grid_spherical(r_edges, theta_edges, phi_edges, fname='amr_grid.inp'):
    with open(fname, 'w') as f:
        f.write("1\n")               # format
        f.write("0\n")               # regular grid
        f.write("100\n")             # spherical coordinates
        f.write("0\n")               # no grid info
        f.write("1 1 1\n")           # all directions active
        f.write(f"{len(r_edges)-1} {len(theta_edges)-1} {len(phi_edges)-1}\n")
        for arr in (r_edges, theta_edges, phi_edges):
            for val in arr:
                f.write(f"{val:.8e}\n")
                
def make_cell_edges(xc):
    dx = np.diff(xc)
    xe = np.zeros(len(xc) + 1)
    xe[1:-1] = 0.5 * (xc[:-1] + xc[1:])
    xe[0] = xc[0] - dx[0] / 2
    xe[-1] = xc[-1] + dx[-1] / 2
    return xe

def write_density_spherical(rho):
    with open('dust_density.inp', 'w') as f:
        f.write("1\n")
        f.write(f"{rho.size}\n")
        f.write("1\n")
        rho.ravel(order='F').tofile(f, sep='\n', format="%13.6e")
        f.write('\n')


def plot_bipolar_r_theta_slice(rho_sph, phi_value=0.0, output='density_rtheta_bipolar.png'):
	"""
	Plot a bipolar (r, theta) slice at phi and phi + pi, with negative r for the back side.

	Parameters:
	-----------
	rho_sph : ndarray
		3D density array (nr, ntheta, nphi)
	r : ndarray
		Radial coordinates (nr,)
	theta : ndarray
		Polar angle coordinates (ntheta,) in radians
	phi : ndarray
		Azimuthal angle array (nphi,) in radians
	phi_value : float
		Azimuthal angle (in radians) for the front side
	output : str
		Filename to save the output plot
	"""
	phi_value = phi_value % (2 * np.pi)
	phi_plus_pi = (phi_value + np.pi) % (2 * np.pi)

	idx_front = np.argmin(np.abs(phi - phi_value))
	idx_back = np.argmin(np.abs(phi - phi_plus_pi))

	# Extract slices
	rho_front = rho_sph[:, :, idx_front]  # (nr, ntheta)
	rho_back = rho_sph[:, :, idx_back]    # (nr, ntheta)
     
	# Find theta of max density at each radius (in degrees - 90)
	theta_deg = theta * 180 / np.pi - 90.0
	theta_max_front = theta_deg[np.argmax(rho_front, axis=1)]
	theta_max_back  = theta_deg[np.argmax(rho_back, axis=1)]

	# Build r arrays
	R, T = np.meshgrid(r / au, theta * 180 / np.pi - 90.0, indexing='ij')        # (nr, ntheta)
	Rneg, Tneg = np.meshgrid(-r / au, theta * 180 / np.pi -90.0, indexing='ij') # mirrored R

	# Plot
	fig, ax = plt.subplots(figsize=(10, 5))
	c1 = ax.pcolormesh(Rneg, Tneg, np.log10(rho_back + 1e-30), cmap='inferno', shading='auto', vmin=-22, vmax=-15.0)
	c2 = ax.pcolormesh(R, T, np.log10(rho_front + 1e-30), cmap='inferno', shading='auto', vmin=-22, vmax=-15.0)

	
	ax.plot(-r / au, theta_max_back, 'c--', lw=1.0)
	ax.plot(r / au, theta_max_front, 'c--', lw=1.0)


	plt.axvline(46.77, color='r', linewidth=1)
	plt.axvline(-46.77, color='r', linewidth=1)
	ax.set_xlabel('Radius: r [au]')
	ax.set_ylabel(r'$\theta$ [deg]')
	cb = fig.colorbar(c1, ax=ax)
	cb.set_label('log Density [g/cm³]')
	ax.set_ylim([-35., 35.])
	plt.tight_layout()
	plt.savefig(output, dpi=150)
	plt.show()

def write_gas_velocity(vxyz, fname="gas_velocity.inp"):
	"""
	Writes the gas velocity field to gas_velocity.inp for RADMC-3D.

	Parameters
	----------
	vxyz : ndarray
		Shape (nr, ntheta, nphi, 3), velocity in cm/s (Cartesian components).
	fname : str
		Output filename (default: 'gas_velocity.inp').
	"""
	nr, ntheta, nphi, _ = vxyz.shape
	nrcells = nr * ntheta * nphi

	# Reshape to (nrcells, 3) and then flatten to (nrcells * 3,)
	vflat = vxyz.reshape(-1, 3).T.flatten()

	with open(fname, 'w') as f:
		f.write('1\n')  # ASCII format
		f.write(f'{nrcells}\n')
		np.savetxt(f, vflat, fmt="%.9e")

import numpy as np
import matplotlib.pyplot as plt

def find_structure_surface(rho_sph, threshold=1e-20, output='structure_faceon.png'):
	"""
	Find and plot the face-on projection of the structure surface where rho > threshold.

	Parameters
	----------
	rho_sph : ndarray
		3D density array with shape (nr, ntheta, nphi)
	r : ndarray
		Radial grid (nr,)
	theta : ndarray
		Polar angle grid in radians (ntheta,)
	phi : ndarray
		Azimuthal angle grid in radians (nphi,)
	threshold : float
		Density threshold to define the surface
	output : str
		Filename to save the face-on plot
	"""
	nr, ntheta, nphi = rho_sph.shape
	x_list, y_list, theta_list = [], [], []
		
	# Precompute θ index of max density for each (r, φ)
	imax_theta = np.argmax(rho_sph, axis=1)  # shape: (nr, nphi)

	for iphi in range(nphi):
		phi_val = phi[iphi]
		for itheta in range(ntheta):
			# Skip if this θ is less than the max density θ for *all* r at this φ
			if not np.any(itheta >= imax_theta[:, iphi]):
				continue

			for ir in range(nr):
				if rho_sph[ir, itheta, iphi] > threshold:
					
					r_val = r[ir]
					theta_val = theta[itheta]

					# Spherical to Cartesian (face-on view)
					sin_theta = np.sin(theta_val)
					x = r_val * sin_theta * np.cos(phi_val)
					y = r_val * sin_theta * np.sin(phi_val)

					if itheta >= np.amax(imax_theta[:min(ir+1,ntheta-1), iphi]):
						x_list.append(x)
						y_list.append(y)
						theta_list.append(theta_val)

						break  # ← crucial! break r-loop after first match


	# Convert lists to arrays
	x_arr = np.array(x_list)
	y_arr = np.array(y_list)
	theta_arr = np.array(theta_list)

	# Plotting
	plt.figure(figsize=(6, 6))
	sc = plt.scatter(1e3*x_arr / au /150.0, 1e3*y_arr / au/150.0, c=np.degrees(theta_arr), s=5, cmap='viridis', alpha=0.8)
	sc = plt.scatter(1e3*x_arr / au /150.0, 1e3*y_arr / au/150.0, c=np.degrees(theta_arr), s=5, cmap='viridis', alpha=0.8)
	plt.xlabel('x [mas]')
	plt.ylabel('y [mas]')
	plt.xlim([-600., 600.0])
	plt.ylim([-600., 600.0])
	plt.title(f'Structure Surface (ρ > {threshold:.1e})')
	plt.colorbar(sc, label=r'$\theta$ [deg]')
	#plt.axis('equal')
	#plt.grid(True)
	plt.tight_layout()
	plt.savefig(output, dpi=150)
	plt.show()


def run():

	print("Computing warped density in spherical coordinates...")
	rho_sph, v_cart = compute_density_warped()

	"""

	print("Generating Cartesian grid and interpolating...")
	xc = np.linspace(-sizex, sizex, nx)
	yc = np.linspace(-sizey, sizey, ny)
	zc = np.linspace(-sizez, sizez, nz)
	xg, yg, zg = np.meshgrid(xc, yc, zc, indexing='ij')

	xi = compute_cell_walls(xc)
	yi = compute_cell_walls(yc)
	zi = compute_cell_walls(zc)
	rho_cart = interpolate_to_cartesian(rho_sph, r, theta, phi, xg, yg, zg)
	print("Plotting density slice...")
	plot_density_slice(xg, zg, rho_cart)


	write_amr_grid(xi, yi, zi)

	write_density(rho_cart)"""
	print('Mapping out surface structure...')
	#find_structure_surface(rho_sph)
	#find_structure_surface(rho_sph, threshold=1e-19)
	#find_structure_surface(rho_sph, threshold=1e-18)
	#find_structure_surface(rho_sph, threshold=1e-17)
     
	print("Plotting density slices...")
	#plot_bipolar_r_theta_slice(rho_sph,phi_value=0.0)
	#plot_bipolar_r_theta_slice(rho_sph,phi_value=np.pi/4.)
	#plot_bipolar_r_theta_slice(rho_sph,phi_value=np.pi/2.)

	print("Writing spherical grid...")
	write_amr_grid_spherical(r_edges, theta_edges, phi_edges)



	print("Writing density...")
	write_density_spherical(rho_sph)
	inc_gas=False
		
	if inc_gas:
		print('Writing velocity...')
		write_gas_velocity(v_cart)
		print('Writing CO density...')
		write_co_number_density(rho_sph, abundance=1e-5)

		print('Writing line input...')
		write_line_input()


	lam = write_wavelength_grid()
	write_stars(lam)

	print('Writing opacity input...')
	write_opacity_control()

	print('Writing radmc3d input...')
	write_radmc3d_inp()
		

	 
if __name__ == '__main__':
	run()
