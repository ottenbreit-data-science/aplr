import setuptools, sys

extra_compile_args = []
if "win" not in sys.platform:
    extra_compile_args.append("-std=c++17")
    extra_compile_args.append("-pthread")

sfc_module = setuptools.Extension(
    name="aplr_cpp",
    sources=["cpp/pythonbinding.cpp"],
    include_dirs=["cpp", "dependencies/eigen-3.4.0", "dependencies/pybind11/include"],
    language="c++",
    extra_compile_args=extra_compile_args,
)

setuptools.setup(
    name="aplr",
    version="7.3.0",
    description="Automatic Piecewise Linear Regression",
    ext_modules=[sfc_module],
    author="Mathias von Ottenbreit",
    author_email="ottenbreitdatascience@gmail.com",
    long_description="Build predictive and interpretable parametric regression or classification machine learning models in Python based on the Automatic Piecewise Linear Regression methodology developed by Mathias von Ottenbreit.",
    long_description_content_type="text/markdown",
    packages=["aplr"],
    install_requires=["numpy>=1.20"],
    python_requires=">=3.8",
    classifiers=["License :: OSI Approved :: MIT License"],
    license="MIT",
    platforms=["Windows", "Linux"],
    url="https://github.com/ottenbreit-data-science/aplr",
)
