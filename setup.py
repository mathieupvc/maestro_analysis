import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='maestro_analysis',
    version='0.0.1',
    author='Mathieu PVC',
    author_email='mathieu.pham-van-cang@pasteur.fr',
    description='Python library based on MNE for MEG/EEG preprocessing and analysis of Maestro experiment',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://gitlab.pasteur.fr/acclab/maestro_analysis',
    license='MIT',
    packages=['maestro_analysis'],
    python_requires='>=3.7',
    install_requires=[
        'mne',
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'seaborn',
        'mido',
        'pingouin',
        'scikit-learn',
    ],
)
