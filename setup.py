from setuptools import setup, find_packages

setup(
    name="active_inference_forager",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pydantic",
    ],
)
