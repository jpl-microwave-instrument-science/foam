from setuptools import setup, find_packages

setup(
    name='FOAM', 
    version='0.1.0',
    author='Alex B. Akins',
    author_email='alexakins@gmail.com',
    packages=find_packages(),
    package_dir = {'foam': 'foam'},
    license='MIT',
    description='The Forward Ocean/Atmosphere Microwave Radiative Transfer Model',
    long_description=open('README.md').read(),
    python_requires='>=3.8',
    install_requires=[
        'numpy >= 1.20.1',
        'scipy >= 1.7.1',
        'pandas >= 1.2.4',
        'xarray >= 0.19.0',
        'dask >= 2021.0.0',
        'numba >= 0.55.0',
        'matplotlib >= 3.4.2',
        'cartopy >= 0.19.0',
        'cftime >= 1.5.0',
        'spiceypy >= 4.0.1',
        'netCDF4 >= 1.5.0',
        'h5py >= 2.10.0',
        'sphinx >= 4.0.2', 
        'tqdm >= 4.0.0'
        ]
    )