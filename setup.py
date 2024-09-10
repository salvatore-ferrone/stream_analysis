from setuptools import setup, find_packages

setup(
    name='stream_analysis',
    version='0.1',
    packages=find_packages(),
    description='intended to perform analysis',
    author='Salvatore Ferrone',
    author_email='salvatore.ferrone@uniroma1.it',
    keywords='python',
    install_requires=[
        "numpy"
    ],
    package_data={
        # If any package contains *.yaml files, include them:
        'gcs': ['*.yaml'],
        # And if you need to be more specific, you can specify the package name:
        # 'gcs': ['paths.yaml'],
    },
)