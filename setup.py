from setuptools import setup, find_packages

setup(
    name="egohos_wrapper",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "numpy"
    ],
    author="Jaskirat Kalra",
    description="A wrapper and post-processor for EgoHOS to fix mask leakages and support frame skipping.",
    python_requires=">=3.7",
)
