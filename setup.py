from setuptools import setup, find_packages

VERSION = '1.0.0'
DESCRIPTION = 'Package for solving geodesy problems'
LONG_DESCRIPTION = 'A package that allows to solve many common geodesic problems.'

# Setting up
setup(
    name="grzymala",
    version=VERSION,
    author="geodeta",
    author_email="jte22466@nezid.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[
    'numpy',
    'matplotlib'
    ],
    keywords=['python', 'geodesy', 'forward problem', 'inverse problem', 'coordinates', 'hirvonen'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License"
    ]
)