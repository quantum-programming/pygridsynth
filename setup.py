from setuptools import setup, find_packages

setup(
    entry_points={
        'console_scripts': [
            'pygridsynth = pygridsynth.__main__:main',
        ],
    }
)
