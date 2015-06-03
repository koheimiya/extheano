from setuptools import setup, find_packages
setup(
        name='extheano',
        version="0.1.0",
        packages=find_packages(),
        description="A small package to provide easy access to Theano, a scientific library for efficient computations",
        url="https://github.com/koheimiya/extheano",
        author="Kohei Miyaguchi",
        author_email="quote.curly@gmail.com",
        license='MIT License',
        classifiers=[
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Programming Language :: Python :: 2.7',
            'Topic :: Utilities',
            'License :: OSI Approved :: MIT License',
            ]
        )
