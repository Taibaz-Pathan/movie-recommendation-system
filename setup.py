from setuptools import setup, find_packages

setup(
    name="movie-recsys",
    version="0.1.0",
    author="Taibaz Pathan",
    description="Movie Recommendation System Using Collaborative Filtering",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy==2.2.3",
        "pandas==2.2.3",
        "scipy==1.15.2",
        "scikit-learn==1.6.1",
        "scikit-surprise==1.1.5",
        "matplotlib==3.9.4",
        "seaborn==0.13.2",
        "pyyaml==6.0.2",
        "tqdm==4.67.1",
    ],
)
