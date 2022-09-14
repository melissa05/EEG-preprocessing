from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='EEG-preprocessing',
    version='1.0',
    packages=[''],
    url='https://github.com/melissa05/EEG-preprocessing',
    license='BSD 3-Clause License',
    author='Melissa Lajtos, Giulia Pezzutti',
    author_email='m.lajtos@student.tugraz.at',
    description='EEG preprocessing for signals contained in .xdf files',
    install_requires=required
)
