import os
import pathlib
import setuptools
from setuptools import find_namespace_packages


requirements_path = pathlib.Path(__file__).resolve().parent.parent / 'requirements.txt'
setuptools.setup(
    name='dream2assist',
    version='0.0.1',
    description='Dream2Assist: An MBRL Framework for Assisting Human Drivers',
    author='jonathan.decastro, thomas.balch, guy.rosman',
    license='CC BY-NC 4.0',
    # long_description=pathlib.Path('README.md').read_text(),
    # long_description_content_type='text/markdown',
    packages=find_namespace_packages(),
    include_package_data=True,
    install_requires=requirements_path.read_text().splitlines(),
    python_requires='>=3.10',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: CC BY-NC 4.0',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
