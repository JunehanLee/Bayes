from setuptools import setup, find_packages

setup(
    name="price_optimize",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "pandas",
        "pymc",
        "arviz",
        "pytensor",
        "matplotlib",
        "scipy",
    ],
)
