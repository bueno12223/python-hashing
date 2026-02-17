from setuptools import setup, Extension

densedict_module = Extension(
    'densedict',
    sources=['densedict.c'],
    extra_compile_args=['-O3', '-march=native', '-std=c11', '-Wall'],
)

setup(
    name='densedict',
    version='1.0.0',
    description='Memory-efficient hash table with Elastic Hashing',
    author='Elastic Hashing PoC',
    ext_modules=[densedict_module],
    python_requires='>=3.7',
)
