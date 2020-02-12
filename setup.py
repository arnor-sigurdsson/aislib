import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


setup(
    name="aislib",
    version="0.1.0",
    url="git@github.com:arnor-sigurdsson/aislib.git",
    license="MIT",
    author="Arnor Ingi Sigurdsson",
    author_email="arnor.sigurdsson@cpr.ku.dk",
    description="Various code I snippets / functions I have found useful.",
    long_description=read("README.md"),
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "tqdm",
        "numpy",
        "torch>=1.0.0",
        "sympy",
        "sklearn",
        "pandas-plink",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
    ],
)
