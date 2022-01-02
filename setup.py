import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "sim-bug-tools-gossq",
    version = "0.0.1",
    author = "Quentin Goss",
    author_email = "gossq@my.erau.edu",
    description = "A toolkit for exploring bugs in software simulations.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/AkbasLab/sim-bug-tools",
    project_urls = {
        "Bug Tracker": "https://github.com/AkbasLab/sim-bug-tools/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires = ">= 3.9"
)