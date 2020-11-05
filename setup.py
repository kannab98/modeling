import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="modeling", # Replace with your own username
    version="0.0.1",
    author="Kirill Ponur",
    author_email="ponur0kirill@yandex.ru",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kannab98/my-pkg",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL License", 
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)