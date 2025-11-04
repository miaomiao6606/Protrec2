from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="protrec2",
    version="1.0.0",
    author="Weijia Kong, Wilson Wen Bin Goh, Limsoon Wong",
    author_email="wilsongoh@ntu.edu.sg",
    description="Tissue-Specific Network-based Missing Protein Recovery Method",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/miaomiao6606/Protrec2",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
    ],
    entry_points={
        "console_scripts": [
            "protrec2=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["data/*.csv", "data/*.tsv", "data/*.txt", "example/*.csv"],
    },
)