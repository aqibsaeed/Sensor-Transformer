from setuptools import setup, find_packages

with open("README.md") as readme_file:
    README = readme_file.read()

setup_args = dict(
    name="sensortransformer",
    version="0.1.6",
    description="Transformer Network for Time-Series and Wearable Sensor Data",
    long_description_content_type="text/markdown",
    long_description=README,
    license="MIT",
    packages=find_packages(),
    author="Aaqib Saeed",
    author_email="aqibsaeed@protonmail.com",
    keywords=["Transformer", "Attention-Mechanism", "Neural Network", "Time-Series", "Sensors"],
    url="https://github.com/aqibsaeed/Sensor-Transformer",
    download_url="https://pypi.org/project/sensor_transformer/"
)

install_requires = [
    "tensorflow>=2.4",
    "einops"
]

if __name__ == "__main__":
    setup(**setup_args, install_requires=install_requires)