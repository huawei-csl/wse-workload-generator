
from setuptools import setup, find_packages

setup(
    name="wse-workload",  # Replace with your package name
    version="0.1.0",  # Initial version
    author="Ahmet Caner Yuzuguler",
    author_email="ahmet.yuzuguler@huawei.com",
    description="A trace generator for WSE workload simulations",
    packages=find_packages(),  # Automatically find sub-packages
    python_requires=">=3.6",  # Specify Python version compatibility
    install_requires=[
        # Add your dependencies here, e.g., "numpy>=1.21.0"
    ],
)
