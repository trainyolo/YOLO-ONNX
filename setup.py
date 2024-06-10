from setuptools import setup

with open("README.md", "rb") as f:
    long_description = f.read().decode("utf-8")

setup(
    name="yolo-onnx",
    version="0.0.1",
    author="trainyolo",
    author_email="info@trainyolo.com",
    description="YOLO onnx runtime",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/trainyolo/YOLO-ONNX",
    install_requires=[
    ],
    packages=[
        "yolo_onnx"
    ],
    classifiers=["Programming Language :: Python :: 3", "License :: OSI Approved :: MIT License"],
    python_requires=">=3.6",
)
