from setuptools import setup, find_packages

setup(
    name="mathbrain",
    version="0.1.0",
    description="Neuroscience-inspired online sequence prediction with complementary learning systems",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Yuyue Li",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "mlx": ["mlx>=0.5.0"],
        "torch": ["torch>=2.0.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
