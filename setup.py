from setuptools import setup, find_packages

setup(
    name="poseguard",
    version="1.0.0",
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
