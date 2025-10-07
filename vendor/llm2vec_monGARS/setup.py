from pathlib import Path
from setuptools import setup, find_packages

package_name = "llm2vec"
version: dict[str, str] = {}
with open(Path(__file__).parent / package_name / "version.py", encoding="utf-8") as fp:
    exec(fp.read(), version)

with open(Path(__file__).parent / "README.md", encoding="utf-8") as fp:
    long_description = fp.read()

setup(
    name=package_name,
    version=version["__version__"],
    author="McGill NLP",
    author_email="parishad.behnamghader@mila.quebec",
    url=f"https://github.com/McGill-NLP/{package_name}",
    description=f"The official {package_name} library",
    python_requires=">=3.8",
    packages=find_packages(include=[f"{package_name}*"]),
    install_requires=[
        "numpy",
        "tqdm",
        "torch",
        "peft",
        "transformers>=4.43.1,<5.0.0",
        "datasets",
        "evaluate",
        "scikit-learn",
    ],
    extras_require={
        "evaluation": ["mteb>=1.14.12"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    zip_safe=False,
)
