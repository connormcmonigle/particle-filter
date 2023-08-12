from skbuild import setup

__version__ = "1.0.0"

base_setup_options = {
    "name": "particle_filter",
    "version": __version__,
    "author": "Connor McMonigle",
    "author_email": "connormcmonigle@gmail.com",
    "description": "A native module implementing CUDA accelerated particle filters.",
    "long_description": "",
    "zip_safe": False,
    "packages": ['particle_filter'],
    "package_data": {'particle_filter': ['__init__.pyi']},
    "package_dir": {'': 'src'},
    "python_requires": ">=3.8",
}

additional_native_setup_options = {
    "cmake_install_dir": 'src/particle_filter',
}

setup(
    **base_setup_options,
    **additional_native_setup_options,
)
