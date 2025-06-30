from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator
from radmc3dPy import analyze
from matplotlib import pyplot as plt

def fit_and_smooth_temperature(x_cart, y_cart, temp_map, AU, R_MIN, R_MAX, fwhm_arcsec, distance_pc, nbins=100):
    # Prepare trimmed x/y and temperature arrays
    iinc_x = np.abs(x_cart) / AU < R_MAX
    iinc_y = np.abs(y_cart) / AU < R_MAX
    x_trim = x_cart[iinc_x]
    y_trim = y_cart[iinc_y]
    temp_map_trim = temp_map[np.where(iinc_x)[0]][:, np.where(iinc_y)[0]]
    
    # Construct 2D radius grid
    XX, YY = np.meshgrid(x_trim, y_trim, indexing='ij')
    RR = np.sqrt(XX**2 + YY**2)
    
    # Fit power-law to radial bins
    rvals = RR.ravel()
    tvals = temp_map_trim.ravel()

    r_bins = np.logspace(np.log10(rvals.min() + 1e-10), np.log10(rvals.max()), nbins + 1)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    t_avg = np.array([
        np.nanmean(tvals[(rvals >= r_bins[i]) & (rvals < r_bins[i+1])])
        if np.any((rvals >= r_bins[i]) & (rvals < r_bins[i+1])) else np.nan
        for i in range(nbins)
    ])

    rmin = R_MIN * AU
    rmax = R_MAX * AU
    valid = (r_centers >= rmin) & (r_centers <= rmax) & (~np.isnan(t_avg))
    r_fit = r_centers[valid]
    t_fit = t_avg[valid]

    def powerlaw(R, T0, q):
        return T0 * (R / r_fit[0]) ** (-q)

    popt, _ = curve_fit(powerlaw, r_fit, t_fit, p0=[t_fit[0], 0.5])
    T0_fit, q_fit = popt

    # Compute model and residuals
    T_model = powerlaw(RR, T0_fit, q_fit)
    T_resid = temp_map_trim - T_model
    T_resid[(RR / AU > R_MAX) | (RR / AU < R_MIN)] = np.nan

    # Apply beam smoothing
    fwhm_au = fwhm_arcsec * distance_pc
    dx = np.abs(x_trim[1] - x_trim[0]) / AU
    dy = np.abs(y_trim[1] - y_trim[0]) / AU
    pixel_size = 0.5 * (dx + dy)
    sigma_pix = fwhm_au / (2.355 * pixel_size)

    valid_mask = (RR / AU >= R_MIN) & (RR / AU <= R_MAX)
    T_resid_filled = np.where(valid_mask, T_resid, 0.0)
    weight_mask = valid_mask.astype(float)

    T_blur = gaussian_filter(T_resid_filled, sigma=sigma_pix)
    W_blur = gaussian_filter(weight_mask, sigma=sigma_pix)

    with np.errstate(invalid='ignore', divide='ignore'):
        T_smoothed = T_blur / W_blur
        T_smoothed[W_blur == 0] = np.nan

    T_smoothed[RR / AU > R_MAX] = np.nan

    plot_fit = False  # Set to False to skip plotting
    if plot_fit:
        r_plot = np.logspace(np.log10(r_fit.min()), np.log10(r_fit.max()), 500)
        plt.figure()
        plt.plot(r_fit / AU, t_fit, 'o', label='Binned average')
        plt.plot(r_plot / AU, powerlaw(r_plot, *popt), '-', label=fr'Fit: $T_0$ = {T0_fit:.1f} K, $q$ = {q_fit:.2f}')
        plt.xlabel('Radius [AU]')
        plt.ylabel('Temperature [K]')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.title('Temperature profile and power-law fit')
        plt.grid(True, which='both', ls=':', lw=0.5)
        plt.tight_layout()
        plt.show()


        plt.pcolormesh(x_trim / AU, y_trim / AU, T_resid[::-1, :], cmap='PuOr', shading='auto') #, vmin=-13, vmax=13)
        plt.colorbar(label='Residual Temperature [K]')
        plt.xlabel('X [AU]')
        plt.ylabel('Y [AU]')
        plt.title('Residual Temperature Map')
        plt.tight_layout()
        plt.show()
        plt.pcolormesh(x_trim / AU, y_trim / AU, T_smoothed[::-1, :], cmap='viridis', shading='auto')
        plt.colorbar(label='Smoothed Residual Temperature [K]')
        plt.xlabel('X [AU]')
        plt.ylabel('Y [AU]')
        plt.title('Smoothed Residual Temperature Map')
        plt.tight_layout()
        plt.show()
        # Return results

    return {
        'x': x_trim,
        'y': y_trim,
        'RR': RR,
        'T_model': T_model,
        'T_resid': T_resid,
        'T_resid_smoothed': T_smoothed,
        'fit_params': (T0_fit, q_fit),
        'r_fit': r_fit,
        't_fit': t_fit
    }



def extract_midplane_temperature(x_cart, y_cart, z_cart, rho_cart, temp_cart, save_path='tmap_midplane.npy'):
    """Extracts temperature at the maximum-density cell along z for each x-y and saves the result."""
    idx_rho_max = np.argmax(rho_cart, axis=2)
    T_midplane = np.take_along_axis(temp_cart, idx_rho_max[:, :, None], axis=2).squeeze()
    
    np.save(save_path, T_midplane)
    print(f"[INFO] Saved midplane temperature map to {save_path}")
    return T_midplane

def load_or_compute_temperature_components():
    if not os.path.isfile('tmap_co.npy') or not os.path.isfile('xy_surf.npy') or not os.path.isfile('z_surf.npy'):
        print("[INFO] Reading dust data and computing CO temperature surface...")
        dust = analyze.readData(ddens=True)
        dust.readDustTemp()

        r, theta, phi = dust.grid.x, dust.grid.y, dust.grid.z
        phi[-1] = 2. * np.pi
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

        # --- CO surface ---
        dz = np.abs(z_cart[1] - z_cart[0])
        N_H = np.cumsum(rho_cart[:, :, ::-1] * dz / (2.3 * 1.67e-24), axis=2)
        N_CO = N_H * 1e-6
        mask = N_CO > 1e15
        idx = np.argmax(mask, axis=2)
        z_emit = z_cart[::-1][idx]
        temp_emit = np.take_along_axis(temp_cart[:, :, ::-1], idx[:, :, None], axis=2).squeeze()

        np.save('xy_surf.npy', np.array([x_cart, y_cart]))
        np.save('z_surf.npy', z_emit)
        np.save('tmap_co.npy', temp_emit)
        np.save('idsurf_map.npy', idx)

        # Optional: save midplane temperature
        extract_midplane_temperature(x_cart, y_cart, z_cart, rho_cart, temp_cart)

    else:
        print("[INFO] Loading precomputed temperature and emission surfaces...")
        temp_emit = np.load('tmap_co.npy')
        x_cart, y_cart = np.load('xy_surf.npy')
        z_emit = np.load('z_surf.npy')
        idx = np.load('idsurf_map.npy')
        
        if os.path.exists('tmap_midplane.npy'):
            T_midplane = np.load('tmap_midplane.npy')
        else:
            print("[WARNING] Midplane temperature file not found. You may need to recompute.")
            T_midplane = None

    return temp_emit, x_cart, y_cart, z_emit, idx



def extract_dT(file_path):
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

    x_vals = np.array(x_vals) 
    y_vals = np.array(y_vals) 
    residuals = np.array(residuals)
    return x_vals, y_vals, radii, residuals