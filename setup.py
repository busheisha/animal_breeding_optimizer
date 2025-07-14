from setuptools import setup, find_packages

setup(
    name="animal_breeding_optimizer",
    version="0.1.0",
    description="Микро-библиотека для оптимизации разведения животных",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="busheisha",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "networkx",
        "deap"
    ],
    python_requires=">=3.7",
    url="https://github.com/busheisha/animal_breeding_optimizer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 