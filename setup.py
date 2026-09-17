from setuptools import setup, find_packages  # type: ignore[reportMissingModuleSource]
from typing import List

def get_requirements(file_path:str)->List:
    '''
    This file will return list of requirements
    '''
    HYPHEN_E_DOT = '-e .'
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('/n','') for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements



setup(
name='MLPROJECT',
version= '0.0.1',
author='Ekangsh',
author_email='sekangsh@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')

)