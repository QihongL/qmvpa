from setuptools import setup, find_packages

setup(name='qmvpa',
      version='0.2',
      description='my MVPA package',
      keywords='neuroimaging, machine learning',
      url='https://github.com/QihongL/qmvpa',
      author='Qihong Lu',
      author_email='lvqihong1992@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'scipy',
      ],
      zip_safe=False)
