============
bp-k-means
============

.. image:: https://img.shields.io/badge/python-3.13+-blue.svg
    :target: https://www.python.org/downloads/
    :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
    :alt: License

A comprehensive boilerplate for Python projects with modern CI/CD setup, testing, documentation, and development tools.

🎯 Project Description
---------------------

This project template provides a solid foundation for Python development with pre-configured:

Development Environment
~~~~~~~~~~~~~~~~~~~~~~~

- **Modern Python Setup**: Python 3.13+ with virtual environment configuration
- **Code Quality Tools**: Pre-commit hooks, linting with Ruff, formatting with Black
- **Testing Framework**: Pytest with coverage reporting and configuration
- **Documentation**: Sphinx-based documentation with automatic API generation

CI/CD Infrastructure
~~~~~~~~~~~~~~~~~~~~

- **GitHub Actions**: Automated testing, linting, and deployment workflows
- **Docker Support**: Containerized development and deployment environment
- **Code Quality**: Automated code quality checks and test coverage reporting
- **Release Management**: Automated versioning and release processes

Project Structure
~~~~~~~~~~~~~~~~~

- **Modular Architecture**: Clean separation of concerns with standard Python package structure
- **Configuration Management**: Centralized configuration with pyproject.toml
- **Documentation**: Complete documentation setup with Sphinx and reStructuredText
- **Testing Setup**: Comprehensive testing configuration with pytest and coverage

🚀 Key Features
--------------

Developer Experience
~~~~~~~~~~~~~~~~~~~~

- **Pre-commit Hooks**: Automatic code formatting and quality checks
- **Modern Tooling**: Latest Python development tools and best practices
- **IDE Configuration**: VS Code settings and task configurations included
- **Environment Management**: Conda and pip-tools support for dependency management

Quick Start
-----------
1. Create an environment in github called `main` and set the following features:
    - Required reviewers: my-name
    - Allow admins to bypass: disabled
    - Deployment branches and tags: main
    - Environment secrets:
        - ``ADMIN_TOKEN``: Administration and actions (read and write)
2. Set the following secrets in your repository settings:
    - ``PAT_TOKEN``: Content and Pull Requests (read and write)
    - ``PYPI_API_TOKEN``: Your PyPI token
    - ``TEST_PYPI_API_TOKEN``: Your Test PyPI token
3. Modify the `.github/workflows/configure_repo.yml` file to set up the minimum (an maximum) python versions, and a list of topics
4. Run the `configure_repo` workflow manually from the Actions tab
5. Set up read the docs to build documentation for this project
6. Clone your repository and start coding!

📚 Documentation
---------------

- 📦 `Installation Guide <docs/installation.rst>`_ - Setup instructions and requirements
- 🤝 `Contributing Guidelines <CONTRIBUTING.rst>`_ - Development standards and contribution process
- 📄 `License <LICENSE.txt>`_ - License terms and usage rights
- 👥 `Authors <AUTHORS.rst>`_ - Project contributors and maintainers
- 📜 `Changelog <CHANGELOG.rst>`_ - Project history and version changes
- 📜 `Code of Conduct <CODE_OF_CONDUCT.rst>`_ - Guidelines for participation and conduct
