import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="t2ebm",
    version="0.0.1",
    author="Sebastian Bordt, Ben Lengerich, Harsha Nori, Rich Caruana",
    author_email="sbordt@posteo.de",
    description="A Natural Language Interface to Explainable Boosting Machines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/interpretml/TalkToEBM",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": ""},
    packages=["t2ebm"],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "matplotlib",
        "tiktoken",
        "guidance",
        "openai",
        "tenacity",
        "scipy",
        "interpret",
    ],
)
