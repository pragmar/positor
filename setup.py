# packaging, remember to rev version
# python3 ./setup.py sdist bdist_wheel
# twine upload dist/positor-0.X.X-py3-none-any.whl

from setuptools import setup
import re

def version() -> str:
    version_pattern = re.compile('__version__\s*=\s*"(\d+\.\d+\.\d+)"')
    with open('./positor/__init__.py') as init_file:
        results = version_pattern.findall(init_file.read())
        if len(results) == 0:
            raise RuntimeError("__init__.py, could not locate semantic version.")
        return results[0]

def read_me() -> str:
    with open('README.md', 'r') as f:
        return f.read()

setup(
    name="positor",
    version=version(),
    description="Utilities for digital archives.",
    long_description=read_me(),
    long_description_content_type='text/markdown',
    python_requires=">=3.9",
    author="pragmar",
    url="https://github.com/pragmar/positor",
    license="MIT",
    packages=['positor'],
    entry_points = {
        "console_scripts": ['positor = positor.positor:main']
    },
    install_requires=[
      "torch==1.11.0", # latest torch has issues with whisper
      "colorama>=0.4.0", 
      "whisper.ai>=1.0.0"
    ],
    include_package_data=False
)
