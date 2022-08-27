from setuptools import setup
from setuptools import find_packages

def get_readme():
    with open("README.md") as readme_file:
        return readme_file.read()

setup(
    name="fibs-reporter",
    version="0.2.0",
    description="Automatically generate a pdf report containing feature importance, " 
                "baseline modelling, spurious correlation detection, and more, from a single command line input",
    url="https://github.com/m-barker/fibs-reporter/",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "fibs = src.main:main"
        ]
    },
    install_requires=["fpdf2==2.5.5",
                      "pandas==1.4.3",
                      "matplotlib==3.5.2",
                      "numpy==1.23.0",
                      "scipy==1.8.1",
                      "Pillow==9.1.1",
                      "opensmile==2.4.1",
                      "audiofile==1.1.0",
                      "sklearn==0.0",
                      "scikit-learn==1.1.1",
                      "seaborn==0.11.2",
                      "pydub==0.25.1",
                      "xgboost==1.6.1"],

    author="Matt Barker",
    license="CC BY-NC 4.0",
    python_requires=">=3.7"
)
