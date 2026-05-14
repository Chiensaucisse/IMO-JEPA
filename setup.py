"""Setup configuration for DITTO package."""

from setuptools import setup, find_packages

setup(
    name='ditto',
    version='0.1.0',
    description='DITTO: Offline Imitation Learning with World Models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/ditto-jepa',
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.21.0',
        'tensorboard>=2.10.0',
        'tqdm>=4.62.0',
        'imageio>=2.9.0',
        'pillow>=8.3.0',
        'einops>=0.6.0',
        'transformers>=4.20.0',
        'stable-worldmodel>=0.1.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'isort>=5.10.0',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
