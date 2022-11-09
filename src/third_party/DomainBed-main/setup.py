from distutils.core import setup
from setuptools import find_packages

#try: # for pip >= 10
#        from pip._internal.req import parse_requirements
#except ImportError: # for pip <= 9.0.3
#        from pip.req import parse_requirements
#
#install_reqs = parse_requirements('requirements.txt', session=True)
#reqs = [str(ir.req) for ir in install_reqs]

setup(
    name="domainbed",
    version=0.1,
    #package_data={p: ["*"] for p in find_packages()},
    package_dir={"":"domainbed"},
    packages=find_packages("domainbed/"),
    url="",
    license="",
    install_requires=[],
    python_requires=">=3.8.0",
    author="",
    author_email="",
    description="",
)
