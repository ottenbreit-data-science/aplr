import setuptools, sys
from pathlib import Path

extra_compile_args = []
extra_link_args = []
if "darwin" in sys.platform:
    extra_compile_args.append("-stdlib=libc++")
    extra_compile_args.append("-Xpreprocessor")
    extra_compile_args.append("-std=c++17")
    extra_compile_args.append("-pthread")
    extra_link_args.append("-stdlib=libc++")
elif "win" not in sys.platform:
    extra_compile_args.append("-std=c++17")
    extra_compile_args.append("-pthread")

sfc_module = setuptools.Extension(
    name="aplr_cpp",
    sources=["cpp/pythonbinding.cpp"],
    include_dirs=["cpp", "dependencies/eigen-3.4.0", "dependencies/pybind11/include"],
    language="c++",
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name="aplr",
    version="10.7.2",
    description="Automatic Piecewise Linear Regression",
    ext_modules=[sfc_module],
    author="Mathias von Ottenbreit",
    author_email="ottenbreitdatascience@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["aplr"],
    install_requires=["numpy>=1.11"],
    python_requires=">=3.8",
    classifiers=["License :: OSI Approved :: MIT License"],
    license="MIT",
    platforms=["Windows", "Linux", "MacOS"],
    url="https://github.com/ottenbreit-data-science/aplr",
)
