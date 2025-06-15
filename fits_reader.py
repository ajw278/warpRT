import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import simple_norm



# --- Configuration ---
filename = "HD__36112_SPHERE_2020-12-26_K.fits"
pixel_size_mas = 12.25  # mas
pixel_size_arcsec = pixel_size_mas / 1000.0  # convert to arcsec

# --- Load FITS file ---
with fits.open(filename) as hdul:
	hdul.info()  # Print to identify PI extension

	print(len(hdul), dir(hdul))
	# You can manually check extension name or index, e.g., 'PI' or 4
	if 'PI' in [h.name for h in hdul]:
		pi_data = hdul['PI'].data
	else:
		# Try a likely extension index if not named
		pi_data = hdul[0].data[-1]  # adjust index as needed
print(pi_data.shape)
# --- Create spatial axes ---
ny, nx = pi_data.shape
x = (np.arange(nx) - nx // 2) * pixel_size_arcsec
y = (np.arange(ny) - ny // 2) * pixel_size_arcsec

# --- Plot ---
plt.figure(figsize=(6, 5))
norm = simple_norm(pi_data, 'sqrt', percent=99.9, vmin=0.03)
plt.imshow(pi_data[::-1,:], origin='lower', extent=[x[-1]*1e3, x[0]*1e3, y[0]*1e3, y[-1]*1e3], norm=norm, cmap='inferno')
plt.xlabel("RA offset [arcsec]")
plt.ylabel("Dec offset [arcsec]")
plt.colorbar(label="Intensity [arb. units]")
plt.xlim([650., -650.])
plt.ylim([650., -650.])
plt.tight_layout()
plt.show()
