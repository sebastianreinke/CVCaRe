from setuptools import setup, find_packages

# Read the requirements.txt file
with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

setup(
    name="CVCaRe",  
    version="7.0",  
    author="Sebastian Reinke",
    author_email="cvcare@sreinke.slmail.me",
    description="Evaluation software for cyclic voltammograms",
    long_description=open('README.md').read(),  
    long_description_content_type="text/markdown",
    url="https://git.noc.rub.de/sreinke/cvcare",
    packages=find_packages(),
    install_requires=required_packages,  # Include the list of dependencies
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',  # Specify the Python version requirement
)