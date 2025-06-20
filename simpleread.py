"""
SIMPLEREAD: A simple Python reading tool for RADMC-3D

By C.P. Dullemond and A. Juhasz (2020)

This Python script contains a few very basic functions for reading the input
and output files of RADMC-3D. They are mostly meant for pedagogical purpose: to
get a feeling for the structure of these files, and for those who just started
with RADMC-3D. These functions only read the ascii versions of these files, not
the binary versions. Also, they are very incomplete, and work only for some very
simple models (e.g. no octree grids, only simple dust opacities, no line data
files).

Usage: In a model directory, in Python (use Python 3):

  from radmc3d_tools.simpleread import *

You can then read, for instance, the dust temperature computed by RADMC-3D
using the radmc3d mctherm command:

  d = read_dusttemp()

Then d contains d.grid and d.dusttemp, the latter being the array of dust
temperatures. There are various read_***() functions: have a look in this
script file.

For real scientific use, it is recommended to use the radmc3dPy package, which is
also included in the RADMC-3D distribution. It is located here (starting from the
root of the RADMC-3D package):

  python/radmc3dPy/

"""
from __future__ import print_function
from mpl_setup import *
try:
    import numpy as np
except ImportError:
    np = None
    print(' Numpy cannot be imported ')
    print(' To use the python module of RADMC-3D you need to install Numpy')
    print(traceback.format_exc())

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print('Warning')
    print('matplotlib.pyplot cannot be imported')
    print('Without matplotlib you can use the python module to set up a model but you will not be able to plot things')
    print('or display images')

import glob

class simplereaddataobject(object):
    """
    Generic data object for the RADMC-3D simpleread.py functions.
    """
    def __init__(self,datatype):
        self.datatype = datatype

class dustkappaobject(object):
    """
    Dust opacity object for the RADMC-3D simpleread.py functions.
    """
    def __init__(self):
        self.datatype = 'dustkappa'
        self.species  = None
    def plot(self,ax=None,color=None):
        """
        Make a log-log plot of this opacity.

        ARGUMENTS:
          ax          The axis object (None means make new subplot)
          color       The color (None means automatic)
        """
        if ax is None:
            fig,ax = plt.subplots()
        if color is None:
            ax.loglog(self.wav,self.kappa_abs,label='abs')
            ax.loglog(self.wav,self.kappa_sca,label='scat')
        else:
            ax.loglog(self.wav,self.kappa_abs,color=color,label=self.species)
            ax.loglog(self.wav,self.kappa_sca,':',color=color)
        ax.set_xlabel(r'$\lambda\;[\mu m]$')
        ax.set_ylabel(r'$\kappa\;[cm^2/g]$')
        return fig,ax

def read_grid():
    """
    Reading the amr_grid.inp file, but only for regular grids (not for octree ones). 

    ARGUMENTS:
      None

    RETURNS:
      Data object containing:

        .nx          Nr of cells in x direction (in spherical coordinates this is the r-direction)
        .ny          Nr of cells in y direction (in spherical coordinates this is the theta-direction)
        .nz          Nr of cells in z direction (in spherical coordinates this is the phi-direction)
        .nxi etc     The same as .nx etc, but now +1, giving the number of cell interfaces
        .crd_sys     String indicating the coordinate system ('car' or 'sph')
        .x,.y,.z     The x, y and z grid cell center locations
        .xi,.yi,.zi  The x, y and z grid cell interface locations (each array 1 longer than the x, y, z ones)

    """
    grid  = simplereaddataobject('grid')
    fname = 'amr_grid.inp'
    print('Reading '+fname)
    data  = np.fromfile(fname, count=-1, sep=" ", dtype=np.float64)

    # Read the header
    hdr   = np.array(data[:10], dtype=np.int_)
    data  = data[10:]

    # Check the file format
    if hdr[0] != 1:
        msg = 'Unknown format number in amr_grid.inp'
        raise RuntimeError(msg)
    if hdr[2] < 100:
        grid.crd_sys = 'car'
    elif (hdr[2] >= 100) & (hdr[2] < 200):
        grid.crd_sys = 'sph'
    else:
        raise ValueError('Unsupported coordinate system identification in the ' + fname + ' file.')

    # Get active dimensions
    grid.act_dim = hdr[4:7]

    # Get the number of cells in each dimensions
    grid.nx = hdr[7]
    grid.ny = hdr[8]
    grid.nz = hdr[9]
    grid.nxi, grid.nyi, grid.nzi = grid.nx + 1, grid.ny + 1, grid.nz + 1

    # Get the cell interfaces
    grid.xi = data[:grid.nxi]
    data = data[grid.nxi:]
    grid.yi = data[:grid.nyi]
    data = data[grid.nyi:]
    grid.zi = data[:grid.nzi]

    # Compute the cell centers
    if grid.crd_sys == 'car':
        grid.x = (grid.xi[0:grid.nx] + grid.xi[1:grid.nx + 1]) * 0.5
        grid.y = (grid.yi[0:grid.ny] + grid.yi[1:grid.ny + 1]) * 0.5
        grid.z = (grid.zi[0:grid.nz] + grid.zi[1:grid.nz + 1]) * 0.5
    else:
        grid.x = np.sqrt(grid.xi[0:grid.nx] * grid.xi[1:grid.nx + 1])
        grid.y = (grid.yi[0:grid.ny] + grid.yi[1:grid.ny + 1]) * 0.5
        grid.z = (grid.zi[0:grid.nz] + grid.zi[1:grid.nz + 1]) * 0.5

    # Now return the grid object
    return grid
    
def read_dustdens(indexorder='fortran'):
    """
    Reading the dust_density.inp file, but only for regular grids, and only
    for text data format (not binary).

    ARGUMENTS:
      indexorder        If 'fortran' then converting array to fortran 
                        index order (default). Else use Python/C order.

    RETURNS:
      Data object containing:

        .grid           A grid object (see read_grid())
        .rhodust        An array with the dust density values (in g/cm^3)
    """
    grid      = read_grid()
    dustdens  = simplereaddataobject('dust_density')
    dustdens.grid = grid
    fname     = 'dust_density.inp'
    print('Reading '+fname)
    data      = np.fromfile(fname, count=-1, sep=" ", dtype=np.float64)
    
    # Read the header
    hdr       = np.array(data[:3], dtype=np.int_)
    data      = data[3:]

    # Check the file format
    if hdr[0] != 1:
        msg = 'Unknown format number in dust_density.inp'
        raise RuntimeError(msg)

    # Get the number of cells, and check against
    nrcells = grid.nx*grid.ny*grid.nz
    if(hdr[1]!=nrcells):
        msg = 'Number of grid cells in dust_density.inp inconsistent with amr_grid.inp'
        raise RuntimeError(msg)

    # Get the number of dust species
    dustdens.nrspec = hdr[2]

    # Convert the rest of the data to the proper shape
    data = np.reshape(data, [dustdens.nrspec, grid.nz, grid.ny, grid.nx])

    # If indexorder is set to 'fortran', then the inner index of the array
    # should be left (even though in Python the inner index is right). This
    # is to assure that the index order in the Python arrays is the same as
    # in the RADMC-3D code. But by setting indexorder to anything else, you
    # can keep Python natural order (which is equal to C index order), in
    # which the inner index is the rightmost index.
    if indexorder=='fortran':
        data = np.swapaxes(data, 0, 3)
        data = np.swapaxes(data, 1, 2)

    # Now add this to the object
    dustdens.rhodust = data

    # Return the dustdens object
    return dustdens

def read_dusttemp(indexorder='fortran'):
    """
    Reading the dust_temperature.dat file, but only for regular grids, and only
    for text data format (not binary).

    ARGUMENTS:
      indexorder        If 'fortran' then converting array to fortran 
                        index order (default). Else use Python/C order.

    RETURNS:
      Data object containing:

        .grid           A grid object (see read_grid())
        .dusttemp       An array with the dust temperature values (in K)
    """
    grid      = read_grid()
    dusttemp  = simplereaddataobject('dust_temperature')
    dusttemp.grid = grid
    fname     = 'dust_temperature.dat'
    print('Reading '+fname)
    data      = np.fromfile(fname, count=-1, sep=" ", dtype=np.float64)
    
    # Read the header
    hdr       = np.array(data[:3], dtype=np.int_)
    data      = data[3:]

    # Check the file format
    if hdr[0] != 1:
        msg = 'Unknown format number in dust_temperature.inp'
        raise RuntimeError(msg)

    # Get the number of cells, and check against
    nrcells = grid.nx*grid.ny*grid.nz
    if(hdr[1]!=nrcells):
        msg = 'Number of grid cells in dust_temperature.inp inconsistent with amr_grid.inp'
        raise RuntimeError(msg)

    # Get the number of dust species
    dusttemp.nrspec = hdr[2]

    # Convert the rest of the data to the proper shape
    data = np.reshape(data, [dusttemp.nrspec, grid.nz, grid.ny, grid.nx])

    # If indexorder is set to 'fortran', then the inner index of the array
    # should be left (even though in Python the inner index is right). This
    # is to assure that the index order in the Python arrays is the same as
    # in the RADMC-3D code. But by setting indexorder to anything else, you
    # can keep Python natural order (which is equal to C index order), in
    # which the inner index is the rightmost index.
    if indexorder=='fortran':
        data = np.swapaxes(data, 0, 3)
        data = np.swapaxes(data, 1, 2)

    # Now add this to the object
    dusttemp.dusttemp = data

    # Return the dusttemp object
    return dusttemp

def read_image(indexorder='fortran'):
    """
    Reading the image.out file.

    ARGUMENTS:
      indexorder        If 'fortran' then converting array to fortran 
                        index order (default). Else use Python/C order.

    RETURNS:
      Data object containing:

        .freq           Frequency at which the image is taken
        .image          An array with the image intensity in erg/(s.cm^2.Hz.ster)
    """
    pc        = 3.08572e18     # Parsec                  [cm]
    cc        = 2.99792458e10  # Light speed             [cm/s]
    image     = simplereaddataobject('image')
    fname     = 'image.out'
    print('Reading '+ fname)
    with open(fname, 'r') as rfile:
        dum = ''

        # Format number
        iformat = int(rfile.readline())

        # Nr of pixels
        dum = rfile.readline()
        dum = dum.split()
        image.nx = int(dum[0])
        image.ny = int(dum[1])

        # Nr of frequencies
        image.nfreq = int(rfile.readline())
        image.nwav = image.nfreq

        # Pixel sizes
        dum = rfile.readline()
        dum = dum.split()
        image.sizepix_x = float(dum[0])
        image.sizepix_y = float(dum[1])

        # Wavelength of the image
        image.wav = np.zeros(image.nwav, dtype=np.float64)
        for iwav in range(image.nwav):
            image.wav[iwav] = float(rfile.readline())
        image.freq = cc / image.wav * 1e4

        # Read the rest of the data
        data = np.fromfile(rfile, count=-1, sep=" ", dtype=np.float64)
        
    # Convert the rest of the data to the proper shape
    if iformat == 1:
        # We have a normal total intensity image
        image.stokes = False
        data = np.reshape(data, [image.nfreq, image.ny, image.nx])
        if indexorder=='fortran':
            data = np.swapaxes(data, 0, 2)

    elif iformat == 3:
        # We have the full stokes image
        image.stokes = True
        data = np.reshape(data, [image.nfreq, 4, image.ny, image.nx])
        if indexorder=='fortran':
            data = np.swapaxes(data, 0, 3)
            data = np.swapaxes(data, 1, 2)

    else:
        msg = 'Unknown format number in image.out'
        raise ValueError(msg)

    # Add this to the object
    image.image = data
    
    # Conversion from erg/s/cm/cm/Hz/ster to Jy/pixel
    conv = image.sizepix_x * image.sizepix_y / pc**2. * 1e23
    image.imageJyppix = image.image * conv
    
    # Create the x and y axes in units of cm
    image.x = ((np.arange(image.nx, dtype=np.float64) + 0.5) - image.nx / 2) * image.sizepix_x
    image.y = ((np.arange(image.ny, dtype=np.float64) + 0.5) - image.ny / 2) * image.sizepix_y
    
    # Return object
    return image

def read_spectrum(dpc=1.):
    """
    Reading the spectrum.out file.

    ARGUMENTS:
      dpc               Distance of observer in parsec (default=1)

    RETURNS:
      Data object containing:

        .wav            Wavelength array of the spectrum in micron
        .freq           Frequency array of the spectrum in Hertz
        .fnu            An array with the flux F_nu at dpc parsec in erg/(s.cm^2.Hz)
    """
    cc        = 2.99792458e10  # Light speed             [cm/s]
    spectrum  = simplereaddataobject('spectrum')
    fname     = 'spectrum.out'
    print('Reading '+ fname)
    with open(fname, 'r') as rfile:
        # Read the format number
        iformat = int(rfile.readline())

        # Read the number of wavelengths
        spectrum.nwav = int(rfile.readline())
        spectrum.freq = spectrum.nwav

        # Read a blank line
        dum = rfile.readline()

        # Read the rest of the data
        data = np.fromfile(rfile, count=-1, sep=" ", dtype=np.float64)

    # Reshape the spectrum
    data          = np.reshape(data, [spectrum.nwav, 2])
    spectrum.wav  = data[:,0]
    spectrum.freq = 1e4 * cc / spectrum.wav
    spectrum.fnu  = data[:,1]

    # Rescale spectrum to other distance
    spectrum.fnu /= dpc**2

    # Return the spectrum
    return spectrum

def read_dustkappa(species=None):
    """
    Reading a dust opacity file (but only the basic dustkappa_*.inp type,
    not the dustkapscatmat_*.inp type).

    ARGUMENTS:
      species           The dust species: Reading dustkappa_<species>.inp
                        If unspecified, read_dustkappa will search for such a file.
                        If it finds a single one, it will read that. Otherwise it
                        will request you to specify species.

    RETURNS:
      Data object containing:

        .wav            Wavelength array of the spectrum in micron
        .freq           Frequency array of the spectrum in Hertz
        .kappa_abs      Absorption opacity in cm^2/gram-of-dust
        .kappa_sca      Scattering opacity in cm^2/gram-of-dust
        .kappa_g        The g-coefficient (between -1 and 1) for non-isotropic scattering
    """
    cc        = 2.99792458e10  # Light speed             [cm/s]

    # Find which dust opacity to read
    if species is None:
        fnames = glob.glob('dustkappa_*.inp')
        if len(fnames)==0:
            msg = 'No file of type dustkappa_*.inp is found in this directory.'
            raise RuntimeError(msg)
        if len(fnames)>1:
            msg = 'More than one file of type dustkappa_*.inp is found in this directory. Please specify the name of the dust species as keyword species.'
            raise RuntimeError(msg)
        species = fnames[0]
    if species[0:10]=='dustkappa_': species = species[10:]
    if species[-4:]=='.inp': species = species[:-4]

    # Read that dust opacity
    #dustkappa = simplereaddataobject('dustkappa')
    dustkappa = dustkappaobject()
    dustkappa.species = species
    fname     = 'dustkappa_'+species+'.inp'
    print('Reading '+ fname)
    with open(fname, 'r') as rfile:
        # Check the file format (skipping comments)
        iformat_str = rfile.readline()
        while iformat_str[0]=='#':
            iformat_str = rfile.readline()
        iformat = int(iformat_str)

        # Read the number of wavelength points
        dustkappa.nwav = int(rfile.readline())
        dustkappa.freq = dustkappa.nwav

        # Now read the rest of the data
        data = np.fromfile(rfile, count=-1, sep=" ", dtype=np.float64)

    # Reshape the data
    if iformat==1:
        data = np.reshape(data, [dustkappa.nwav, 2])
    elif iformat==2:
        data = np.reshape(data, [dustkappa.nwav, 3])
    elif iformat==3:
        data = np.reshape(data, [dustkappa.nwav, 4])
    else:
        msg = 'Format number of kappa file not known'
        raise RuntimeError(msg)

    # Extract the information
    dustkappa.wav        = data[:,0]
    dustkappa.freq       = 1e4 * cc / dustkappa.wav
    dustkappa.kappa_abs  = data[:,1]
    if iformat>1:
        dustkappa.kappa_sca = data[:,2]
    if iformat>2:
        dustkappa.kappa_g = data[:,3]

    # Return dustkappa
    return dustkappa

def read_gastemp(indexorder='fortran'):
    """
    Reading the gas_temperature.inp file, but only for regular grids, and only
    for text data format (not binary).

    ARGUMENTS:
      indexorder        If 'fortran' then converting array to fortran
                        index order (default). Else use Python/C order.

    RETURNS:
      Data object containing:

        .grid           A grid object (see read_grid())
        .gastemp        An array with the gas temperature values (in K)
    """
    grid      = read_grid()
    gastemp   = simplereaddataobject('gas_temperature')
    gastemp.grid = grid
    fname     = 'gas_temperature.inp'
    print('Reading '+fname)
    data      = np.fromfile(fname, count=-1, sep=" ", dtype=np.float64)

    # Read the header
    hdr       = np.array(data[:2], dtype=np.int_)
    data      = data[2:]

    # Check the file format
    if hdr[0] != 1:
        msg = 'Unknown format number in gas_temperature.inp'
        raise RuntimeError(msg)

    # Get the number of cells, and check against
    nrcells = grid.nx*grid.ny*grid.nz
    if(hdr[1]!=nrcells):
        msg = 'Number of grid cells in gas_temperature.inp inconsistent with amr_grid.inp'
        raise RuntimeError(msg)

    # Convert the rest of the data to the proper shape
    data = np.reshape(data, [grid.nz, grid.ny, grid.nx])

    # If indexorder is set to 'fortran', then the inner index of the array
    # should be left (even though in Python the inner index is right). This
    # is to assure that the index order in the Python arrays is the same as
    # in the RADMC-3D code. But by setting indexorder to anything else, you
    # can keep Python natural order (which is equal to C index order), in
    # which the inner index is the rightmost index.
    if indexorder=='fortran':
        data = np.swapaxes(data, 0, 2)

    # Now add this to the object
    gastemp.gastemp = data

    # Return the gastemp object
    return gastemp

def read_gasvelocity(indexorder='fortran'):
    """
    Reading the gas_velocity.inp file, but only for regular grids, and only
    for text data format (not binary).

    ARGUMENTS:
      indexorder        If 'fortran' then converting array to fortran
                        index order (default). Else use Python/C order.

    RETURNS:
      Data object containing:

        .grid           A grid object (see read_grid())
        .gasvelo        An array with the gas velocity values (in cm/s)
    """
    grid      = read_grid()
    gasvelo   = simplereaddataobject('gas_velocity')
    gasvelo.grid = grid
    fname     = 'gas_velocity.inp'
    print('Reading '+fname)
    data      = np.fromfile(fname, count=-1, sep=" ", dtype=np.float64)

    # Read the header
    hdr       = np.array(data[:2], dtype=np.int_)
    data      = data[2:]

    # Check the file format
    if hdr[0] != 1:
        msg = 'Unknown format number in gas_velocity.inp'
        raise RuntimeError(msg)

    # Get the number of cells, and check against
    nrcells = grid.nx*grid.ny*grid.nz
    if(hdr[1]!=nrcells):
        msg = 'Number of grid cells in gas_velocity.inp inconsistent with amr_grid.inp'
        raise RuntimeError(msg)

    # Convert the rest of the data to the proper shape
    data = np.reshape(data, [grid.nz, grid.ny, grid.nx, 3])

    # If indexorder is set to 'fortran', then the inner index of the array
    # should be left (even though in Python the inner index is right). This
    # is to assure that the index order in the Python arrays is the same as
    # in the RADMC-3D code. But by setting indexorder to anything else, you
    # can keep Python natural order (which is equal to C index order), in
    # which the inner index is the rightmost index.
    if indexorder=='fortran':
        data = np.swapaxes(data, 0, 3)
        data = np.swapaxes(data, 1, 2)

    # Now add this to the object
    gasvelo.velocity = data

    # Return the gasvelo object
    return gasvelo

def read_molnumdens(molecule,indexorder='fortran'):
    """
    Reading a numberdens_xxxx.inp file, but only for regular grids, and only
    for text data format (not binary).

    ARGUMENTS:
      molecule          Then name of the molecule (e.g. 'co')
      indexorder        If 'fortran' then converting array to fortran
                        index order (default). Else use Python/C order.

    RETURNS:
      Data object containing:

        .grid           A grid object (see read_grid())
        .numdens        An array with the number density values of that molecule (in 1/cm^3)
    """
    grid      = read_grid()
    molnumdens   = simplereaddataobject('molecule_number_density')
    molnumdens.grid = grid
    fname     = 'numberdens_'+molecule+'.inp'
    print('Reading '+fname)
    data      = np.fromfile(fname, count=-1, sep=" ", dtype=np.float64)

    # Read the header
    hdr       = np.array(data[:2], dtype=np.int_)
    data      = data[2:]

    # Check the file format
    if hdr[0] != 1:
        msg = 'Unknown format number in '+fname
        raise RuntimeError(msg)

    # Get the number of cells, and check against
    nrcells = grid.nx*grid.ny*grid.nz
    if(hdr[1]!=nrcells):
        msg = 'Number of grid cells in '+fname+' inconsistent with amr_grid.inp'
        raise RuntimeError(msg)

    # Convert the rest of the data to the proper shape
    data = np.reshape(data, [grid.nz, grid.ny, grid.nx])

    # If indexorder is set to 'fortran', then the inner index of the array
    # should be left (even though in Python the inner index is right). This
    # is to assure that the index order in the Python arrays is the same as
    # in the RADMC-3D code. But by setting indexorder to anything else, you
    # can keep Python natural order (which is equal to C index order), in
    # which the inner index is the rightmost index.
    if indexorder=='fortran':
        data = np.swapaxes(data, 0, 2)

    # Now add this to the object
    molnumdens.numdens = data

    # Return the molnumdens object
    return molnumdens

def read_mollevelpop(molecule,indexorder='fortran'):
    """
    Reading a levelpop_xxxx.dat file, but only for regular grids, and only
    for text data format (not binary).

    ARGUMENTS:
      molecule          Then name of the molecule (e.g. 'co')
      indexorder        If 'fortran' then converting array to fortran
                        index order (default). Else use Python/C order.

    RETURNS:
      Data object containing:

        .grid           A grid object (see read_grid())
        .pop            An array with the level population values of that molecule (in units of number density 1/cm^3)
    """
    grid      = read_grid()
    mollevelpop   = simplereaddataobject('molecule_level_populations')
    mollevelpop.grid = grid
    fname     = 'levelpop_'+molecule+'.dat'
    print('Reading '+fname)
    data      = np.fromfile(fname, count=-1, sep=" ", dtype=np.float64)

    # Read the header
    hdr       = np.array(data[:3], dtype=np.int_)
    data      = data[3:]

    # Check the file format
    if hdr[0] != 1:
        msg = 'Unknown format number in '+fname
        raise RuntimeError(msg)

    # Get the number of cells, and check against
    nrcells = grid.nx*grid.ny*grid.nz
    if(hdr[1]!=nrcells):
        msg = 'Number of grid cells in '+fname+' inconsistent with amr_grid.inp'
        raise RuntimeError(msg)

    # Number of levels
    mollevelpop.nlevels = hdr[2]

    # Identities of the levels (for associating them to the levels in the molecule_xxxx.inp file)
    hdr       = np.array(data[:mollevelpop.nlevels], dtype=np.int_)
    data      = data[mollevelpop.nlevels:]
    mollevelpop.levels = hdr

    # Convert the rest of the data to the proper shape
    data = np.reshape(data, [grid.nz, grid.ny, grid.nx, mollevelpop.nlevels])

    # If indexorder is set to 'fortran', then the inner index of the array
    # should be left (even though in Python the inner index is right). This
    # is to assure that the index order in the Python arrays is the same as
    # in the RADMC-3D code. But by setting indexorder to anything else, you
    # can keep Python natural order (which is equal to C index order), in
    # which the inner index is the rightmost index.
    if indexorder=='fortran':
        data = np.swapaxes(data, 0, 3)
        data = np.swapaxes(data, 1, 2)

    # Now add this to the object
    mollevelpop.pop = data

    # For convenience, also compute the relative level populations
    # (which sum up to 1 in each cell)
    mollevelpop.relpop = np.zeros_like(mollevelpop.pop)
    if indexorder=='fortran':
        dum = mollevelpop.pop.sum(axis=0)
        for i in range(mollevelpop.nlevels):
            mollevelpop.relpop[i,:,:,:] = mollevelpop.pop[i,:,:,:]/dum[:,:,:]
    else:
        dum = mollevelpop.pop.sum(axis=3)
        for i in range(mollevelpop.nlevels):
            mollevelpop.relpop[:,:,:,i] = mollevelpop.pop[:,:,:,i]/dum[:,:,:]

    # Return the mollevelpop object
    return mollevelpop

def read_mean_intensity(indexorder='fortran'):
    """
    Reading the mean_intensity.out file, but only for regular grids, and only
    for text data format (not binary).

    ARGUMENTS:
      indexorder        If 'fortran' then converting array to fortran
                        index order (default). Else use Python/C order.

    RETURNS:
      Data object containing:

        .grid           A grid object (see read_grid())
        .meanint        An array with the mean intensity
    """
    grid      = read_grid()
    meanint   = simplereaddataobject('mean_intensity')
    meanint.grid = grid
    fname     = 'mean_intensity.out'
    print('Reading '+fname)
    data      = np.fromfile(fname, count=-1, sep=" ", dtype=np.float64)

    # Read the header
    hdr       = np.array(data[:3], dtype=np.int_)
    data      = data[3:]

    # Get the frequencies
    nwav      = hdr[-1]
    freq      = data[:nwav]
    data      = data[nwav:]
    meanint.freq = freq

    # Check the file format
    if hdr[0] != 2:
        msg = 'Unknown format number in mean_intensity.out'
        raise RuntimeError(msg)

    # Get the number of cells, and check against
    nrcells = grid.nx*grid.ny*grid.nz
    if(hdr[1]!=nrcells):
        msg = 'Number of grid cells in gas_temperature.inp inconsistent with amr_grid.inp'
        raise RuntimeError(msg)

    # Convert the rest of the data to the proper shape
    data = np.reshape(data, [nwav,grid.nz, grid.ny, grid.nx])

    # If indexorder is set to 'fortran', then the inner index of the array
    # should be left (even though in Python the inner index is right). This
    # is to assure that the index order in the Python arrays is the same as
    # in the RADMC-3D code. But by setting indexorder to anything else, you
    # can keep Python natural order (which is equal to C index order), in
    # which the inner index is the rightmost index.
    if indexorder=='fortran':
        data = np.swapaxes(data, 1, 3)

    # Now add this to the object
    meanint.meanint = data

    # Return the meanint object
    return meanint


def read_subbox(name='dust_temperature',indexorder='fortran'):
    """
    Reading a subbox file. Default is subbox for the dust
    temperature. To use it, you first need to create such a
    dataset with RADMC-3D. For instance using the command

      ./radmc3d subbox_dust_temperature subbox_nxyz 64 64 64 subbox_xyz01 -2.e15 2.e15 -2.e15 2.e15 -2.e15 2.e15

    This will create the file dust_temperature_subbox.out,
    which you can then read in python with

      from radmc3d_tools import simpleread as sr
      q = sr.read_subbox(name='dust_temperature')

    ARGUMENTS:
      name              The subbox file to read = name+"_subbox.out"
      indexorder        If 'fortran' then converting array to fortran
                        index order (default). Else use Python/C order.

    RETURNS:
      Data object containing:

        .grid           A grid object (see read_grid())
        .data           An array with the data
    """

    # Create the subbox object and read the basic data
    fname     = name+'_subbox.out'
    subbox    = simplereaddataobject(fname)
    print('Reading '+fname)
    data      = np.fromfile(fname, count=-1, sep=" ", dtype=np.float64)

    # Extract the header and create the main data box
    hdr       = np.array(data[:15], dtype=np.int_)
    data      = data[15:]

    # Get the grid data
    nx        = int(hdr[1])
    ny        = int(hdr[2])
    nz        = int(hdr[3])
    xmin      = hdr[5]
    xmax      = hdr[6]
    ymin      = hdr[7]
    ymax      = hdr[8]
    zmin      = hdr[9]
    zmax      = hdr[10]

    # Create the grid object and link it to the subbox object
    grid      = simplereaddataobject('grid')
    grid.nx   = nx
    grid.ny   = ny
    grid.nz   = nz
    grid.nxi  = nx+1
    grid.nyi  = ny+1
    grid.nzi  = nz+1
    grid.crd_sys = 'car'
    grid.xi   = np.linspace(xmin,xmax,nx+1)
    grid.yi   = np.linspace(ymin,ymax,ny+1)
    grid.zi   = np.linspace(zmin,zmax,nz+1)
    grid.x    = 0.5*(grid.xi[1:]+grid.xi[:-1])
    grid.y    = 0.5*(grid.yi[1:]+grid.yi[:-1])
    grid.z    = 0.5*(grid.zi[1:]+grid.zi[:-1])
    subbox.grid = grid

    # Check the file format
    if hdr[0] != 2:
        msg = 'Unknown format number in '+fname
        raise RuntimeError(msg)

    # Convert the rest of the data to the proper shape
    data = np.reshape(data, [grid.nz, grid.ny, grid.nx])

    # If indexorder is set to 'fortran', then the inner index of the array
    # should be left (even though in Python the inner index is right). This
    # is to assure that the index order in the Python arrays is the same as
    # in the RADMC-3D code. But by setting indexorder to anything else, you
    # can keep Python natural order (which is equal to C index order), in
    # which the inner index is the rightmost index.
    if indexorder=='fortran':
        data = np.swapaxes(data, 0, 2)

    # Now add this to the object
    subbox.data = data

    # Return the subbox object
    return subbox
    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# --- Constants ---
pc = 3.08572e18  # parsec in cm
au = 1.496e+13

#MWC758 distance
dist = 156.0

def plot_image(image, wavelength_index=0, logscale=True, vmin=None, vmax=None, cmap='inferno'):
	"""
	Plot RADMC-3D image in Jy/pixel at given wavelength index.
	"""
	img = image.image[:,:,0]

	print(np.amax(img))
	extent = [image.x[0]*1e3/au/dist, image.x[-1]*1e3/au/dist, image.y[0]*1e3/au/dist, image.y[-1]*1e3/au/dist]

	print(extent)
	fig, ax = plt.subplots(figsize=(6, 5))
	if logscale:
		norm = LogNorm(vmin=vmin or img[img > 0].min(), vmax=vmax or img.max())
	else:
		norm = None
		

	xsp = np.linspace(extent[0],extent[1], img.T.shape[0])
	xg, yg = np.meshgrid(xsp, xsp,indexing='ij')
	
	rg = np.sqrt(xg**2+yg**2)
	coronagraph = 115.0
	img.T[img.T<1e-15] = 1e-15
	img.T[rg<coronagraph] = 1e-15

	from astropy.visualization import simple_norm
	norm = simple_norm(img.T[::-1,:], 'sqrt', percent=99)

	im = ax.imshow(img.T[::-1,:], origin='lower', extent=extent, cmap=cmap, norm=norm)
    
	ax.set_xlabel('RA offset [mas]')
	ax.set_ylabel('Dec offset [mas]')
	ax.set_xlim([650.,-650.0])
	ax.set_ylim([-650., 650.0])
	#lam = image.wav[0]
	#cbar = fig.colorbar(im, ax=ax)
	#cbar.set_label('Intensity [Jy/pixel]')
	plt.tight_layout()
	plt.show()

def plot_multipanel(image):
	import numpy as np
	import matplotlib.pyplot as plt
	from matplotlib.colors import LogNorm
	from astropy.io import fits
	from astropy.visualization import simple_norm
	# ----------------------------------------------------------
	# Load PI FITS image
	fits_filename = "HD__36112_SPHERE_2020-12-26_K.fits"
	pixel_size_mas = 12.25  # in milliarcseconds
	pixel_size_arcsec = pixel_size_mas / 1000.0

	# --- Load FITS file ---
	with fits.open(fits_filename) as hdul:
		hdul.info()  # Print to identify PI extension

		print(len(hdul), dir(hdul))
		# You can manually check extension name or index, e.g., 'PI' or 4
		if 'PI' in [h.name for h in hdul]:
			pi_data = hdul['PI'].data
		else:
			# Try a likely extension index if not named
			pi_data = hdul[0].data[-1]  # adjust index as needed


	noise_region = pi_data

	sigma = np.std(noise_region[np.isfinite(noise_region)])  # robust sigma estimate
	print(f"Estimated noise sigma: {sigma:.3e}")

	# --- Define sigma levels for contours ---
	sigma_levels = np.array([5, 10,  20, 50])
	levels = sigma_levels * sigma

	ny, nx = pi_data.shape
	x_fits = (np.arange(nx) - nx // 2) * pixel_size_mas
	y_fits = (np.arange(ny) - ny // 2) * pixel_size_mas




	# Compute extent in mas
	extent_radmc = [image.x[0]*1e3/au/dist,
					image.x[-1]*1e3/au/dist,
					image.y[0]*1e3/au/dist,
					image.y[-1]*1e3/au/dist]

	print(image.image.shape)
	if len(image.image.shape)==4:
		# Extract Stokes parameters (shape: [nx, ny, nlambda])
		I = image.imageJyppix[:, :, 0, 0]  # Stokes I
		Q = image.imageJyppix[:, :, 1,0]  # Stokes Q
		U = image.imageJyppix[:, :, 2,0]  # Stokes U
		V = image.imageJyppix[:, :, 3,0]  # Stokes V

		# Compute Polarised Intensity
		img = np.sqrt(Q**2 + U**2+V**2)
	else: 
		img = image.image[:, :, 0]
        
	floor= 1.e-15
	img = img.T  # transpose for consistent orientation
	img[img < floor] = floor

	# Apply coronagraph mask
	xsp = np.linspace(extent_radmc[0], extent_radmc[1], img.shape[0])
	xg, yg = np.meshgrid(xsp, xsp, indexing='ij')
	rg = np.sqrt(xg**2 + yg**2)
	img[rg < 100.0] = floor
     

	xg_fits, yg_fits = np.meshgrid(x_fits, y_fits, indexing='ij')
	rg_dat = np.sqrt(xg_fits**2 + yg_fits**2)
	pi_data[rg_dat < 100.0] = floor
    

	extent_fits = [x_fits[-1], x_fits[0], y_fits[0], y_fits[-1]]

	# ----------------------------------------------------------
	# Plot side-by-side
	fig, axs_flip = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
	axs = (axs_flip[1], axs_flip[0])

	# --- Left panel: RADMC-3D image ---
	norm1 = simple_norm(img[::-1, :], 'sqrt', percent=97.0)
	axs[0].imshow(img[::-1, :], extent=extent_radmc, origin='lower', cmap='inferno', norm=norm1)
	axs[0].contour(pi_data, levels=levels, extent=extent_fits, origin='lower', colors='grey', linewidths=1.0)
		
	# Set tick mark color to white
	axs[0].tick_params(axis='both', color='white')
	axs[1].tick_params(axis='both', color='white')

	
	axs[0].set_xlabel("RA offset [mas]")
	axs[1].set_ylabel("Dec offset [mas]")
	axs[0].set_xlim([650, -650])
	axs[0].set_ylim([-650, 650])

	# --- Right panel: PI image ---
	norm2 = simple_norm(pi_data, 'sqrt', percent=99.9)
	axs[1].imshow(pi_data[:,:], extent=extent_fits, origin='lower', cmap='inferno', norm=norm2)
	axs[1].set_xlabel("RA offset [mas]")
	axs[1].set_xlim([650, -650])
	axs[1].set_ylim([-650, 650])

	plt.tight_layout()
	plt.savefig('MWC750_comparison.pdf', bbox_inches='tight', format='pdf')
	plt.show()



def plot_temperature(d):
	import numpy as np
	import matplotlib.pyplot as plt
	from astropy.io import fits

	# --- Physical constants (CGS units) ---
	h = 6.62607015e-27     # Planck constant [erg·s]
	k = 1.380649e-16       # Boltzmann constant [erg/K]
	c = 2.99792458e10      # Speed of light [cm/s]

	# --- Read RADMC-3D image.out ---
	print(dir(d))
	image = d.image  # shape: (nx, ny, nlam) or (nx, ny)
	wavelengths = d.wav * 1e-4  # Convert from micron to cm
	freqs = c / wavelengths

	# --- Convert to brightness temperature ---
	def intensity_to_Tb(I_nu, freq, method='planck'):
		if method == 'planck':
			# Planck brightness temperature
			with np.errstate(divide='ignore', invalid='ignore'):
				Tb = (h * freq) / (k * np.log((2 * h * freq**3 / (c**2 * I_nu)) + 1))
			Tb[np.isnan(Tb)] = 0.0
			Tb[Tb < 0] = 0.0
		elif method == 'rj':
			Tb = (c**2 / (2 * k * freq**2)) * I_nu
		else:
			raise ValueError("method must be 'planck' or 'rj'")
		return Tb

		# --- Convert entire cube to Tb ---
	if image.ndim == 3:
		nx, ny, nlam = image.shape
		Tb_cube = np.zeros_like(image)
		for i in range(nlam):
			Tb_cube[:, :, i] = intensity_to_Tb(image[:, :, i], freqs[i], method='planck')

		# --- Compute peak brightness temperature map ---
		Tb_peak = np.max(Tb_cube, axis=2)

	else:
		# Single-channel image
		nx, ny = image.shape
		Tb_peak = intensity_to_Tb(image, freqs[0], method='planck')

	# --- Plot ---
	plt.imshow(Tb_peak, origin='lower', cmap='inferno')
	plt.colorbar(label='Peak Brightness Temperature [K]')
	title = f"Peak T$_B$ map ({d.nwav} channels)" if d.nwav > 1 else f"T$_B$ at {d.wav[0]:.2f} μm"
	plt.title(title)
	plt.xlabel('x (pixels)')
	plt.ylabel('y (pixels)')
	plt.tight_layout()
	plt.savefig('Tb_peak_map.png', dpi=200)
	plt.show()

	# --- Save FITS file ---
	fits.writeto('Tb_peak_map.fits', Tb_peak, overwrite=True)


def read_Tfits():
	import numpy as np
	import matplotlib.pyplot as plt
	from astropy.io import fits

	from astropy.wcs import WCS

	# --- Load FITS file ---
	filename = "delta_peakintensity_gaussian_data.fits"

	hdul = fits.open(filename)
	data = hdul[0].data
	header = hdul[0].header
	wcs = WCS(header)
	# Slice the data if it's 3D (e.g., nchan × ny × nx) — otherwise use as is
	if data.ndim == 3:
		data = data[0]  # Assume first channel is what you want

	print(data, np.amax(data), np.amin(data))
	#data[np.absolute(data)>1e5] = 0.0

	# --- Plot ---
	plt.figure(figsize=(6, 5))
	ax = plt.subplot(projection=wcs.celestial)
	im = ax.imshow(data, origin='lower', cmap='RdBu_r', vmin=0.5, vmax=1.5)
	plt.colorbar(im, label='$\Delta T_\mathrm{B}$ [K]')

	ax.set_xlabel('Right Ascension (J2000)')
	ax.set_ylabel('Declination (J2000)')

	plt.tight_layout()
	plt.savefig("delta_tb_map.png", dpi=200)
	plt.show()


if __name__ == '__main__':
	

	image = read_image()
	#plot_temperature(image)
	#exit()

	plot_multipanel(image)
	exit()
	plot_image(image, wavelength_index=0, logscale=True, vmin=1e-16, vmax=1e-13)
