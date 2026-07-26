from setuptools import setup, find_packages

setup(
    name="movie-recsys",
    version="0.1.0",
    author="Taibaz Pathan",
    description="Movie Recommendation System Using Collaborative Filtering",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy==1.26.4",
        "pandas==2.2.2",
        "scipy==1.13.0",
        "scikit-learn==1.4.2",
        "scikit-surprise==1.1.3",
        "matplotlib==3.8.4",
        "seaborn==0.13.2",
        "pyyaml==6.0.1",
        "tqdm==4.66.4",
    ],
)
