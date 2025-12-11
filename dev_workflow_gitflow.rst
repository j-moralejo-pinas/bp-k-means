Development Workflow
====================

Creating a Feature Branch
-------------------------

1. Make sure you're on the dev branch and it's up to date:

.. code-block:: bash

    git checkout dev
    git pull

2. Create a new feature branch:

.. code-block:: bash

    git checkout -b feature/your-feature-name

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

This project follows a structured Gitflow branching model to maintain code quality and enable collaborative development.

Branch Types
------------

main
~~~~
- The production-ready branch
- Contains stable, tested code
- Protected branch requiring pull request reviews
- Only accepts merges from ``dev`` or ``hotfix`` branches

dev
~~~
- The integration branch for ongoing development
- Contains the latest development features
- All feature and bugfix branches merge here first
- Regularly merged into ``main`` when stable

feature/\*
~~~~~~~~~~
- Created for new features or enhancements
- Branched from ``dev``
- Naming convention: ``feature/feature-name`` or ``feature/issue-number-description``
- Merged back into ``dev`` via pull request

release/\*
~~~~~~~~~~
- Created for preparing a new production release
- Branched from ``dev``
- Naming convention: ``release/version-number``
- Used for final testing and bug fixes before merging into ``main``

bugfix/\*
~~~~~~~~~
- Created for non-urgent bug fixes
- Branched from ``dev``
- Naming convention: ``bugfix/bug-description`` or ``bugfix/issue-number-description``
- Merged back into ``dev`` via pull request

hotfix/\*
~~~~~~~~~
- Created for urgent production fixes
- Branched from ``main``
- Naming convention: ``hotfix/critical-issue-description``
- Merged directly into ``main`` and then back-merged into ``dev``

meta/\*
~~~~~~~
- Created for non-code changes (documentation, CI/CD, etc.)
- Branched from ``main``
- Naming convention: ``meta/change-description``
- Merged back into ``main`` via pull request

Merge Workflows
---------------

Feature/Bugfix → Dev
~~~~~~~~~~~~~~~~~~~~

1. Rebase ``dev`` into your feature/bugfix branch:

.. code-block:: bash

    git checkout feature/your-feature
    git fetch origin
    git rebase origin/dev

2. Create a pull request from ``feature/your-feature`` to ``dev``
3. Use **merge commit** or **squash and merge** to maintain clean commit history
4. Delete the feature branch after successful merge

Dev → Release → Main
~~~~~~~~~~~~~~~~~~~~

1. When ready for a release, create a release branch from ``dev``:

.. code-block:: bash

    git checkout dev
    git pull origin dev
    git checkout -b release/x.y.z
    git push origin release/x.y.z

2. Perform final testing and bug fixes on the release branch
3. Create a pull request from ``release/x.y.z`` to ``main``
4. Use **squash commit** for a clean release commit
5. After merge, CI pipeline will tag the release, publish packages, deploy documentation and merge ``main`` back into ``dev`` to keep branches synchronized

Hotfix → Main
~~~~~~~~~~~~~

1. Rebase ``main`` into your hotfix branch:

.. code-block:: bash

    git checkout hotfix/critical-fix
    git fetch origin
    git rebase origin/main

2. Create a pull request from ``hotfix/critical-fix`` to ``main``
3. Use **squash and merge** for clean hotfix commits
4. After merge, CI pipeline will tag the hotfix release, publish packages, deploy documentation and merge ``main`` back into ``dev`` to keep branches synchronized

Meta → Main
~~~~~~~~~~~

1. Rebase ``main`` into your meta branch:

.. code-block:: bash

    git checkout meta/your-meta-change
    git fetch origin
    git rebase origin/main

2. Create a pull request from ``meta/your-meta-change`` to ``main``
3. Use **squash and merge** for clean meta commits
4. After merge, CI pipeline will deploy documentation and merge ``main`` back into ``dev`` to keep branches synchronized

Branch Protection Rules
-----------------------

- ``main``: Requires pull request reviews, status checks must pass
- ``dev``: Requires pull request reviews, status checks must pass
- Direct pushes to ``main`` and ``dev`` are prohibited
- All branches must be up-to-date before merging

Workflow Examples
-----------------

**Creating a Feature**

.. code-block:: bash

    # Start from dev
    git checkout dev
    git pull origin dev

    # Create feature branch
    git checkout -b feature/your-feature-name

    # Make changes and commit
    git add .
    git commit -m "feat: implement your-feature-name"

    # Push and create PR
    git push origin feature/your-feature-name

**Preparing for Merge**

.. code-block:: bash

    # Before creating PR, rebase on latest dev
    git fetch origin
    git rebase origin/dev

    # Resolve conflicts if any, then force push
    git push --force-with-lease origin feature/your-feature-name