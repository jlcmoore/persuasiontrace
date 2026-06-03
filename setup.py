"""
/setup.py

Author: Jared Moore
Date: July, 20252
"""

from pathlib import Path

import setuptools


def _read_requirements_file(path: str) -> list[str]:
    """Read pip requirement lines from a file with support for nested includes.

    Parameters:
        path: Relative path to the requirements file.

    Returns:
        A list of requirement specifiers with comments and blank lines removed.
    """
    requirement_path = Path(path)
    if not requirement_path.exists():
        return []

    requirements: list[str] = []
    for raw_line in requirement_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            include_path = line.split(maxsplit=1)[1]
            nested_path = requirement_path.parent / include_path
            requirements.extend(_read_requirements_file(str(nested_path)))
            continue
        requirements.append(line)
    return requirements


core_requirements = _read_requirements_file("requirements-api.txt")
mechanism_rl_requirements = _read_requirements_file("requirements-mechanism-rl.txt")
dev_requirements = _read_requirements_file("requirements-dev.txt")
full_requirements: list[str] = []
for requirement_specifier in (
    *core_requirements,
    *mechanism_rl_requirements,
    *dev_requirements,
):
    if requirement_specifier not in full_requirements:
        full_requirements.append(requirement_specifier)

setuptools.setup(
    name="continuouspersuasion",
    version="0.0.1",
    author="Jared Moore",
    author_email="jared@jaredmoore.org",
    description="Continuous Persuasion",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    package_data={
        "data": ["*"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=core_requirements,
    extras_require={
        "mechanism-rl": mechanism_rl_requirements,
        "dev": dev_requirements,
        "full": full_requirements,
    },
    entry_points={
        "console_scripts": [
            "read_database = api.read_database:read_database",
            "annotate_rounds = annotation.runner:main",
        ]
    },
)
