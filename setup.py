from setuptools import setup, find_packages

setup(
    name="poseguard",
    version="1.0.2",
    author="PoseGuard AI Contributors",
    author_email="contributors@poseguard.io",
    description="PoseGuard AI — 3D Skeletal Animation Motion Defect Analysis System",
    url="https://github.com/A35-777/PoseGaurd-AI",
    packages=find_packages(),
    package_data={
        "poseguard": ["app/static/**/*", "app/static/*"],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "poseguard=poseguard.cli.main:main",
        ],
    },
)
