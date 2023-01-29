from pathlib import Path

from setuptools import setup, find_namespace_packages


style_packages = ["black==22.10.0", "isort==5.10.1", "pylint==2.15.10"]
test_packages = ["pytest>=7.2.0", "pytest-cov==4.0.0"]

ROOT_DIR = Path(__file__).absolute().parent

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()


# Required packages
def list_reqs(*, filename: str = "requirements.txt") -> None:
    with open(ROOT_DIR / filename, encoding="utf-8") as f:
        return f.read().splitlines()


setup(
    name="nlp",
    version="0.1.0",
    description="A simple package by Neidu.",
    author="Chinedu Ezeofor",
    author_email="neidue@email.com",
    packages=find_namespace_packages(),
    url="https://github.com/chineidu/nyc-taxi-price-prediction",
    install_requires=list_reqs(),
    python_requires=">=3.8",
    extras_require={
        "dev": style_packages + test_packages,
        "test": test_packages,
    },
    include_package_data=True,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
)
