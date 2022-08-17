from setuptools import setup
from setuptools import find_packages

setup(
    name="fibs-reporter",
    version="0.1.0",
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
                      "pdfrw==0.4",
                      "opensmile==2.4.1",
                      "audiofile==1.1.0",
                      "sklearn==0.0",
                      "scikit-learn==1.1.1",
                      "statsmodels==0.13.2",
                      "seaborn==0.11.2",
                      "pydub==0.25.1",
                      "xgboost==1.6.1"],

    author="Matt Barker",
    license="CC BY-NC-SA 2.0",
    python_requires=">=3.8"
)
