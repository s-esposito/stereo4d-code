from setuptools import setup, find_packages

setup(
    name="o3d_renderer",
    version="0.1.0",
    description="Open3D-based point cloud renderer for online and offline visualization",
    author="Stefano Esposito",
    packages=find_packages(),
    install_requires=[
        "open3d>=0.17.0",
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.60.0",
    ],
    python_requires=">=3.8",
)
