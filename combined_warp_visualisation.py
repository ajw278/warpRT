import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.gridspec import GridSpec
from pathlib import Path
from mpl_setup import *

from temp_extract import extract_dT

def plot_disc_surfaces(
    name,
    inclination_deg,
    PA_deg,
    warp_profile_file,
    dv_xygrid_file,
    dv_obs_file,
    residuals_file,
    r_max=230.0,
    zboost=1.0,
    elev=30,
    azim=45,
    output_dir="figures",
    vrange=0.3,
    tag=''
):
    # Inclination and PA in radians
    i0 = np.radians(inclination_deg)
    PA0 = np.radians(PA_deg)

    # Functions
    def rotate(x, y, z, inc, PA):
        x1 = x*np.cos(PA) - y*np.sin(PA)
        y1 = x*np.sin(PA) + y*np.cos(PA)
        z1 = z
        x2 = x1
        y2 = y1*np.cos(inc) - z1*np.sin(inc)
        z2 = y1*np.sin(inc) + z1*np.cos(inc)
        return x2, y2, z2

    def flatten_disc(x, y, z, i0, PA0):
        y1 = y*np.cos(-i0) - z*np.sin(-i0)
        z1 = y*np.sin(-i0) + z*np.cos(-i0)
        x1 = x
        x2 = x1*np.cos(-PA0) - y1*np.sin(-PA0)
        y2 = x1*np.sin(-PA0) + y1*np.cos(-PA0)
        z2 = z1
        return x2, y2, z2

    def build_surface(radii, delta_i, delta_PA, radii_grid, phi_grid):
        R_grid, phi_grid = np.meshgrid(radii_grid, phi_grid, indexing='ij')
        i_grid = i0 + np.interp(R_grid, radii, delta_i)
        PA_grid = PA0 + np.interp(R_grid, radii, delta_PA)
        x0 = R_grid * np.cos(phi_grid)
        y0 = R_grid * np.sin(phi_grid)
        z0 = np.zeros_like(x0)
        xw, yw, zw = rotate(x0, y0, z0, i_grid, PA_grid)
        return flatten_disc(xw, yw, zw, i0, PA0)

    # Load warp profile
    radii, delta_i, delta_PA = np.loadtxt(warp_profile_file, unpack=True)

    # Grids
    N_r_poly = 100
    N_phi_poly = 200
    N_r_line = 25
    N_phi_line = 40

    radii_poly = np.linspace(radii.min(), radii.max(), N_r_poly)
    phi_poly = np.linspace(0, 2*np.pi, N_phi_poly)
    radii_line = np.linspace(radii.min(), radii.max(), N_r_line)
    phi_line = np.linspace(0, 2*np.pi, N_phi_line)

    # Surfaces
    x_flat_poly, y_flat_poly, z_flat_poly = build_surface(radii, delta_i, delta_PA, radii_poly, phi_poly)
    x_flat_line, y_flat_line, z_flat_line = build_surface(radii, delta_i, delta_PA, radii_line, phi_line)

    radii_vertices = np.sqrt(x_flat_poly.ravel()**2 + y_flat_poly.ravel()**2)
    r_min = np.amin(radii)

    tri = mtri.Triangulation(x_flat_poly.ravel(), y_flat_poly.ravel())
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

    # Interpolations
    xygrid = np.load(dv_xygrid_file)
    dv_obs = np.load(dv_obs_file)
    interp_dv = RegularGridInterpolator(
        (xygrid[0], xygrid[1]),
        dv_obs.T,
        bounds_error=False,
        fill_value=np.nan
    )
    dv_on_surface = interp_dv(np.column_stack((x_flat_poly.ravel(), y_flat_poly.ravel())))

    x_vals, y_vals, _, residuals = extract_dT(residuals_file)
    interp_T = LinearNDInterpolator(
        np.column_stack((x_vals, y_vals)),
        residuals,
        fill_value=np.nan
    )
    T_on_surface = interp_T(np.column_stack((x_flat_poly.ravel(), y_flat_poly.ravel())))

    # Color normalization
    norm_dv = plt.Normalize(-vrange, vrange)
    cmap_dv = plt.cm.seismic
    facecolors_dv = cmap_dv(norm_dv(dv_on_surface))
    facecolors_dv_valid = facecolors_dv[valid_triangles[:,0]]

    norm_T = plt.Normalize(-5, 20)
    cmap_T = plt.cm.magma
    facecolors_T = cmap_T(norm_T(T_on_surface))
    facecolors_T_valid = facecolors_T[valid_triangles[:,0]]


    # Figure setup
    fig = plt.figure(figsize=(14,7))
    gs = GridSpec(2,2,height_ratios=[0.05,0.95],width_ratios=[1,1],hspace=0.02,wspace=0.02)

    # Colorbars at the top
    cax_dv = fig.add_subplot(gs[0,0])
    cax_T = fig.add_subplot(gs[0,1])

    # Panels
    ax_dv = fig.add_subplot(gs[1,0],projection='3d')
    ax_T = fig.add_subplot(gs[1,1],projection='3d')

    # dv panel
    poly_dv = Poly3DCollection(verts,facecolors=facecolors_dv_valid,edgecolor="none",alpha=1.0)
    ax_dv.add_collection3d(poly_dv)

    # T panel
    poly_T = Poly3DCollection(verts,facecolors=facecolors_T_valid,edgecolor="none",alpha=1.0)
    ax_T.add_collection3d(poly_T)

    # Wireframes
    z_lift = 1e-3
    radii_wireframe = np.sqrt(x_flat_line**2 + y_flat_line**2)
    ax_cols = ["k", "w"]
    iax = 0
    for ax in [ax_dv, ax_T]:
        for i in range(len(x_flat_line)):
            if np.all((radii_wireframe[i,:]<r_min)|(radii_wireframe[i,:]>r_max)):
                continue
            ax.plot(
                x_flat_line[i,:],
                y_flat_line[i,:],
                z_flat_line[i,:]+z_lift,
                color=ax_cols[iax], lw=0.5,
                zorder=10
            )
        for j in range(N_phi_line):
            r_line = radii_wireframe[:,j]
            mask = (r_line>=r_min)&(r_line<=r_max)
            if not np.any(mask):
                continue
            idx_start = np.argmax(mask)
            idx_end = len(r_line)-np.argmax(mask[::-1])
            ax.plot(
                x_flat_line[idx_start:idx_end,j],
                y_flat_line[idx_start:idx_end,j],
                z_flat_line[idx_start:idx_end,j]+z_lift,
                color=ax_cols[iax], lw=0.5,
                zorder=10
            )
        iax+=1
       

        rmax = min(np.amax(x_flat_line), r_max)/1.5
        ax.set_zlim([-rmax/zboost,rmax/zboost])
        ax.set_xlim([-rmax,rmax])
        ax.set_ylim([-rmax,rmax])
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.grid(False)

    # Colorbars with ticks and labels at top
    mappable_dv = plt.cm.ScalarMappable(cmap=cmap_dv, norm=norm_dv)
    cbar_dv = fig.colorbar(mappable_dv, cax=cax_dv, orientation="horizontal")
    cbar_dv.set_label(r"$\Delta v_{\mathrm{LOS}}$ [km/s]")
    cbar_dv.ax.xaxis.set_ticks_position('top')
    cbar_dv.ax.xaxis.set_label_position('top')

    mappable_T = plt.cm.ScalarMappable(cmap=cmap_T, norm=norm_T)
    cbar_T = fig.colorbar(mappable_T, cax=cax_T, orientation="horizontal")
    cbar_T.set_label(r"$\Delta T_\mathrm{B}$ [K]")
    cbar_T.ax.xaxis.set_ticks_position('top')
    cbar_T.ax.xaxis.set_label_position('top')

    fig.suptitle(f"{name}"+tag, fontsize=14)

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save figure
    outpath = Path(output_dir) / f"{name.replace(' ','_')}_elev{elev}_azim{azim}.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {outpath}")

    plt.show()


if __name__=='__main__':

    plot_disc_surfaces(
    name="MWC 758",
    inclination_deg=21.0,
    PA_deg=0.0,# PA_deg=30.0,
    warp_profile_file="mwc758_warpprofile.dat",
    dv_xygrid_file="MWC 758_xygrid.npy",
    dv_obs_file="MWC 758_dv_obs.npy",
    residuals_file="azimuthal_peakintensity_residuals_mwc758.dat",
    r_max=230.0,
    elev=45,
    azim=20,
    zboost=4.0
    )

    plot_disc_surfaces(
    name="HD 34700",
    inclination_deg=35.3,
    PA_deg=0.0,#PA_deg=92.9,
    warp_profile_file="hd34700_warpprofile.dat",
    dv_xygrid_file="HD 34700_xygrid.npy",
    dv_obs_file="HD 34700_dv_obs.npy",
    tag='A',
    residuals_file="azimuthal_peakintensity_residuals_hd34700.txt",
    r_max=490.0,
    zboost=3.0,
    vrange=0.7
    )
