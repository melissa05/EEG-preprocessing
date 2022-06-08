from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='EEG-preprocessing',
    version='1.0',
    packages=[''],
    url='https://github.com/WriessneggerLab/EEG-preprocessing',
    license='BSD 3-Clause License',
    author='Giulia Pezzutti',
    author_email='giulia.pezzutti@studenti.unipd.it',
    description='EEG preprocessing for signals contained in .xdf files',
    install_requires=required
)
