from setuptools import setup, find_packages

setup(
    name='image-classification-app',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A project for image classification using PyTorch or TensorFlow.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch',  # or 'tensorflow' if using TensorFlow
        'pandas',
        'numpy',
        'matplotlib',
        'scikit-learn',
        'jupyter',  # for running notebooks
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)