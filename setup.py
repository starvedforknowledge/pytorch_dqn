from setuptools import setup, find_packages
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name='pytorch_dqn',
    version='0.0.1',
    description='deepq learning with pytorch and openai gym.',
    author='Tyler Dauphinee',
    install_requires=required,
    python_requires='>=3.7',
    packages=find_packages()
)