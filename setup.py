from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE.md') as f:
    license = f.read()

setup(
    name='hdrie',
    version='0.0.1',
    description='Transform images to HDR and render them with several techniques',
    long_description=readme,
    author='Ole Martin Ruud',
    author_email='barskern@outlook.com',
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
    ],
    url='https://github.com/barskern/hdrie',
    license=license,
    packages=find_packages(exclude=('tests'))
)
