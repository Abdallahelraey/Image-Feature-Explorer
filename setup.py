from setuptools import setup, find_packages

setup(
    name="mnist_cnn_project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow==2.17.0",
        "opencv-python==4.10.0.84",
        "matplotlib==3.9.2"
    ],

)