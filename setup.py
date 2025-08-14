from setuptools import setup

setup(
    entry_points={
        "console_scripts": [
            "pygridsynth = pygridsynth.cli:main",
        ],
    }
)
