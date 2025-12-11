Development Workflow
====================

Pulling from Main
-----------------

1. Make sure you're on the main branch and it's up to date:

.. code-block:: bash

    git checkout main
    git pull

Making Changes
--------------

1. Make your changes in the appropriate files
2. Add tests for new functionality
3. Update documentation if needed
4. Run the test suite to ensure everything works

Running During Development
--------------------------

When running code during development, use:

.. code-block:: bash

    PYTHONPATH='/path/to/package-name/src' python your_script.py

Branching Model and Workflow
============================

This project follows Trunk-Based Development to maintain code quality and enable collaborative development.

Branch Types
------------

main
~~~~~~~~
- The production-ready branch
- Contains stable, tested code
- Only accepts merges from ``hotfix`` branches
- Tagged with a new version number on each release

hotfix/\*
~~~~~~~~~
- Created for urgent production fixes
- Branched from a tagged commit in ``main`` or a ``hotfix`` branch
- Naming convention: ``hotfix/critical-issue-description``
- Merged back into ``main`` via pull request
- Tagged with a new version number the previous commit before merging

Merge Workflows
---------------

Main Development Flow
~~~~~~~~~~~~~~~~~~~~~

1. Pull the latest changes from ``main``:
.. code-block:: bash

    git checkout main
    git pull

2. Push your changes to main:
.. code-block:: bash

    git checkout main
    git push

Hotfix → Main
~~~~~~~~~~~~~

1. Create your branch from a tagged commit from ``main`` or an existing ``hotfix`` branch:

.. code-block:: bash

    git checkout -b hotfix/your-branch

2. Create a pull request from ``hotfix/your-branch`` to ``main``
3. Use **merge commit** to keep track of all changes
4. Do not delete the ``hotfix/your-branch`` after merging to keep since that branch will contain the patched release
