"""Setup script for FastKNN Python package"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory.parent / "README_LIBRARY.md").read_text()

setup(
    name="fastknn",
    version="1.0.0",
    author="CMSE 822 Team",
    author_email="",
    description="High-performance K-Nearest Neighbors with OpenMP, MPI, and CUDA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fastknn",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "scikit-learn>=0.24",  # For comparison tests
        ],
    },
    package_data={
        "fastknn": ["*.so", "*.dll", "*.dylib"],
    },
    zip_safe=False,
)
