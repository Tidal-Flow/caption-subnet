

from setuptools import setup, find_packages

setup(
    name='bittensor-subnet',
    version='0.1.0',
    author='Your Name',
    author_email='vemulapallimukesh@gmail.com',
    description='A Bittensor subnet for audio captioning and transcription evaluation.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'bittensor',
        'numpy',
        'scipy',
        'pydub',
        'torch',  # Assuming you're using PyTorch for STT models
        'transformers',  # If using Hugging Face models like Whisper
        'soundfile',  # For audio file handling
        'webrtcvad',  # For voice activity detection
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)