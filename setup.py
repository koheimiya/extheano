from setuptools import setup, find_packages
setup(
        name='extheano',
        version="0.1.1",
        packages=find_packages(),
        description="Small package providing an easy access to Theano, a scientific library for efficient computations",
        url="https://github.com/koheimiya/extheano",
        author="Kohei Miyaguchi",
        author_email="koheimiyaguchi@gmail.com",
        license='MIT License',
        classifiers=[
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3.3',
            'Topic :: Utilities',
            'License :: OSI Approved :: MIT License',
            ]
        )
