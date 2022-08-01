import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sim-bug-tools-gossq",
    version="0.0.1",
    author="Quentin Goss",
    author_email="gossq@my.erau.edu",
    description="A toolkit for exploring bugs in software simulations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AkbasLab/sim-bug-tools",
    project_urls={
        "Bug Tracker": "https://github.com/AkbasLab/sim-bug-tools/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">= 3.9",
    install_requires=[
        "numpy >= 1.21.1",
        "matplotlib >= 3.4.2",
        "scipy >= 1.7.1",
        "scikit-learn >= 0.24.2",
        "networkx >= 2.6.2",
        "pandas >= 1.3.2",
        "openturns >= 1.17",
        "traci >= 1.11",
        "PySimpleGUI >= 4.56.0",
        "treelib >= 1.6.1",
    ],
)
