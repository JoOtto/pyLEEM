import setuptools
import versioneer

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]

setuptools.setup(
    name="pyLEEM",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Johannes Otto",
    author_email="<>",
    description="Analysis library for low energy electron microscopy data with .nlp file import.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JoOtto/pyLEEM",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements,
    entry_points={
        "xarray.backends": ["pyLEEM=pyLEEM.XArrayExt:NLPBackend"],
    },
)

