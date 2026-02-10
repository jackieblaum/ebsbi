from setuptools import setup, find_packages

setup(
    name="ebsbi",  # Replace with your project's name
    version="0.1.0",
    description="A package for inferring parameters of eclipsing binary stars using neural simulation-based inference.",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    author="Jackie Blaum",
    author_email="jackie.blaum@gmail.com",
    url="https://github.com/jackieblaum/ebsbi",  # Replace with your repo
    packages=find_packages(where="src"),  # Use "src" as the base directory
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "numpy",
        "pymc",
        "pyyaml",
        "astropy",
        "phoebe",
        "extinction",
        "isochrones",
        "binarysedfit",
        "nbi",
        "torch",
        "torchvision",
        "torchaudio",
        "torchvision",
        "torchvision",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "black", "flake8"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "ebsbi=ebsbi.main:main",  # Replace with your entry point
        ],
    },
)
