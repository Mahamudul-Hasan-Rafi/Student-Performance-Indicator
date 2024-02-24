from setuptools import find_packages, setup


def get_requirements(file):
    with open(file) as f:
        requirements = f.readlines()
        requirements = [req.replace("/n", "") for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements


setup(
    name="Exam Score Predictor",
    version="0.0.1",
    author="Mahamudul Hasan Rafi",
    author_email="rafimahamudulhasan98@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)
