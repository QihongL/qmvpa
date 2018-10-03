from setuptools import setup, find_packages

setup(name='qmvpa',
      version='0.52',
      description='my MVPA package',
      keywords='neuroimaging, machine learning',
      url='https://github.com/qihongl/qmvpa',
      author='Qihong Lu',
      author_email='lvqihong1992@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'scipy',
          'sklearn',
          'brainiak'
      ],
      zip_safe=False)
