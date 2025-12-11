Installation Guide
==================

This guide provides step-by-step instructions for installing and setting up the bp-k-means project template. Choose the installation section that best fits your needs.

.. contents:: Table of Contents
    :local:
    :depth: 2

Prerequisites
=============

Before installing the project, ensure you have the following requirements:

* **Python 3.13** (required for this project)
* **Git** for cloning the repository
* **Internet connection** for downloading dependencies

User Installation
=================

This section is for users who want to use the project template without modifying the source code.

Quick Start
-----------

1. **Clone the Repository**: Clone the project repository from GitHub

.. code-block::

    git clone https://github.com/j-moralejo-pinas/bp-k-means.git
    cd bp-k-means

2. **Set Up Virtual Environment (Recommended)**: While not mandatory, using a virtual environment is highly recommended to avoid dependency conflicts

.. code-block::

    # Using conda (recommended)
    conda create -n bp-k-means-env python=3.13
    conda activate bp-k-means-env

    # OR using venv
    python -m venv venv
    # On Linux/macOS:
    source venv/bin/activate
    # On Windows:
    venv\Scripts\activate

3. **Install the Package**: Install the project and its dependencies

.. code-block::

    pip install -e .

4. **Verify Installation**: Test that the installation was successful

.. code-block::

    python -c "import bp_k_mean; print('Installation successful!')"

Docker Installation (Alternative)
---------------------------------

If you prefer to use Docker instead of a local Python installation, you can run the project in a containerized environment.

Prerequisites for Docker
~~~~~~~~~~~~~~~~~~~~~~~~

* **Docker** and **Docker Compose** installed on your system
* **Git** for cloning the repository

Docker Setup
~~~~~~~~~~~~

1. **Clone the Repository**

.. code-block::

    git clone https://github.com/j-moralejo-pinas/bp-k-means.git
    cd bp-k-means

2. **Build the Docker Image**: Build the application using Docker Compose. This will create a Docker image with all necessary dependencies pre-installed

.. code-block::

    docker-compose build

3. **Verify Docker Installation**: Test that the Docker setup works

.. code-block::

    docker-compose run --rm app python -c "import bp_k_mean; print('Docker installation successful!')"

**Docker Benefits**

* **Isolated environment** - No conflicts with your system Python
* **Consistent setup** - Same environment across different machines
* **Easy cleanup** - Remove containers when done
* **Pre-configured dependencies** - All system libraries included

Developer Installation
======================

This section is for developers who want to contribute to the project or modify the source code.

Development Setup
-----------------

1. **Clone and Navigate**

.. code-block::

    git clone https://github.com/j-moralejo-pinas/bp-k-means.git
    cd bp-k-means

2. **Set Up Development Environment**: Create a virtual environment (recommended)

.. code-block::

    conda create -n bp-k-means-dev python=3.13
    conda activate bp-k-means-dev

3. **Install in Development Mode**: Install the package with development dependencies
    This installs the project in editable mode with all development tools including:

    * ``pytest`` - Testing framework
    * ``pyright`` - Type checking
    * ``pre-commit`` - Git hooks for code quality
    * ``ruff`` - Fast Python linter and formatter
    * ``pydoclint`` - Documentation linting
    * ``docformatter`` - Documentation formatting
    * ``pytest-cov`` - Test coverage
    * ``pyupgrade`` - Code modernization
    * ``sphinx`` - Documentation generation
    * ``sphinx-autoapi`` - Automatic API documentation generation

.. code-block::

    pip install -e ".[dev,docs]"

4. **Set Up Pre-commit Hooks**: Install pre-commit hooks to ensure code quality

.. code-block::

    pre-commit install

5. **Configure Type Checking**: Link your development environment to Pyright for proper type checking. Create a ``pyrightconfig.local.json`` file in the project root

.. code-block::

    {
        "venvPath": "/path/to/your/conda/envs",
        "venv": "bp-k-means-dev"
    }

.. [#f1] Replace ``/path/to/your/conda/envs`` with your actual conda environments path (e.g., ``/home/username/miniconda3/envs`` or ``/home/username/micromamba/envs``).

6. **Configure Environment**: Set the ``PYTHONPATH`` environment variable or add it to your shell profile to include the source directory (``~/.bashrc``, ``~/.zshrc``, etc.)

.. code-block::

    export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

    Or add this to your shell profile (``~/.bashrc``, ``~/.zshrc``, etc.).

7. **Verify Installation**: Test that the development installation was successful

.. code-block::

    python -c "import bp_k_mean; print('Development installation successful!')"
    pytest --version
    ruff --version
    pyright --version

Troubleshooting
===============

**Common Issues**

**Import Errors**

If you encounter import errors, ensure the ``PYTHONPATH`` is set correctly

.. code-block::

    export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

**Virtual Environment Issues**

If you have issues with virtual environments, try

.. code-block::

    # For conda environments
    conda info --envs  # List all environments
    conda activate bp-k-means-dev  # Activate the environment

    # For venv environments
    which python  # Check which Python you're using
    pip list  # Check installed packages

**Docker Issues**

If Docker commands fail

.. code-block::

    # Check Docker is running
    docker --version
    docker-compose --version

    # Check Docker permissions (Linux)
    sudo usermod -aG docker $USER
    # Then log out and back in

**Getting Help**

* Check the project's GitHub issues: https://github.com/j-moralejo-pinas/bp-k-means/issues
* Review the documentation for detailed usage examples
* Ensure all dependencies are correctly installed

See Also
========

- `Contributing <CONTRIBUTING.rst>`_ - How to contribute to the project
