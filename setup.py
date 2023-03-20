from setuptools import setup, find_packages

setup(
   name='squared_neural_families',
   version='0.1',
   description='Implementation of squared neural families',
   author='Russell Tsuchida',
   author_email='russell.tsuchida@data61.csiro.au',
   packages=find_packages(),  #same as name
   install_requires=['wheel'] #external packages as dependencies
)
