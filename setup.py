
import setuptools
from numpy.distutils.core import setup
from numpy.distutils.core import Extension

extra_compile_args = ['-Wall', '-Wextra', '-pedantic', '-funroll-loops', '-O2']
fortran_source_files = [
    'nestedbasinsampling/sampling/fortran/noguts.f90',
    'nestedbasinsampling/sampling/fortran/galilean.f90',
    'nestedbasinsampling/structure/fortran/hardshell.f90',
    'nestedbasinsampling/nestedsampling/fortran/combineruns.f90']
fortran_modules = [filename.split('.')[0].replace("/", ".")
                   for filename in fortran_source_files]
ext_modules = [Extension(
    mod, sources=[filename], extra_compile_args=extra_compile_args)
    for mod, filename in zip(fortran_modules, fortran_source_files)]

if __name__ == "__main__":
    setup(name                 = 'nestedbasinsampling',
          description          = "Nested Basin Sampling",
          author               = "Matthew Griffiths",
          author_email         = "matthewghgriffiths@gmail.com",
          packages             = setuptools.find_packages(),
          package_data         = {'': ['*.f90']},
          include_package_data = True,
          ext_modules          = ext_modules)
