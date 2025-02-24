#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tasks.py
========
This module defines common development tasks using Invoke.

Run tasks with commands like:
    invoke test
    invoke lint
    invoke format
    invoke tree
    invoke stats
    invoke docker

    invoke build-api
    invoke serve-mkdocs

    invoke demo.examples
    invoke demo.options
    invoke demo.market-env
    invoke demo.stock
"""

import os
import sys
from invoke import task, Collection

IGNORED_PATTERNS = ["__pycache__", ".pyc", ".pytest_cache"]


def use_pty():
    """
    Determine whether to use a pseudo-terminal (pty) based on the operating system.

    On Windows, the 'pty' module is not supported, so this function returns False.
    On other platforms, it returns True.

    Returns
    -------
    bool
        True if a pseudo-terminal should be used; otherwise, False.
    """
    return False if sys.platform == "win32" else True


@task(help={'verbose': "Run tests in verbose mode."})
def test(ctx, verbose=False):
    """
    Run the test suite using pytest.

    Parameters
    ----------
    ctx : invoke.Context
        The invoke context.
    verbose : bool, optional
        If True, run tests in verbose mode (default is False).

    Returns
    -------
    None

    Bash:
        invoke test
    """
    command = "python -m pytest"
    if verbose:
        command += " -v"
    ctx.run(command, pty=use_pty())


@task
def lint(ctx):
    """
    Run lint checks using flake8.

    Parameters
    ----------
    ctx : invoke.Context
        The invoke context.

    Returns
    -------
    None

    Bash:
        invoke lint
    """
    ctx.run("flake8", pty=use_pty())


@task
def format(ctx):
    """
    Format code using Black.

    Parameters
    ----------
    ctx : invoke.Context
        The invoke context.

    Returns
    -------
    None

    Bash:
        invoke format
    """
    ctx.run("black src tests demos", pty=use_pty())


@task
def stats(ctx):
    """
    Run pygount to display a summary of code metrics (lines of code vs comments).

    Parameters
    ----------
    ctx : invoke.Context
        The invoke context.

    Returns
    -------
    None

    Bash:
        invoke stats
    """
    ctx.run("pygount -f summary .", pty=use_pty())


@task
def docker(ctx):
    """
    Build and run the Docker containers using docker-compose.

    This task runs:
        docker-compose up --build

    Parameters
    ----------
    ctx : invoke.Context
        The invoke context.

    Returns
    -------
    None

    Bash:
        invoke docker
    """
    ctx.run("docker-compose up --build", pty=use_pty())


def print_tree(start_path, ignore_patterns, indent=""):
    """
    Recursively print the directory tree, ignoring directories or files that match
    any of the ignore patterns.

    Parameters
    ----------
    start_path : str
        The root directory from which to start printing the tree.
    ignore_patterns : list of str
        Patterns to ignore.
    indent : str, optional
        The indentation to use (default is an empty string).

    Returns
    -------
    None
    """
    try:
        items = sorted(os.listdir(start_path))
    except PermissionError:
        return
    for item in items:
        # Skip items that match any ignore pattern.
        if any(pattern in item for pattern in ignore_patterns):
            continue
        path = os.path.join(start_path, item)
        print(indent + "|-- " + item)
        if os.path.isdir(path):
            print_tree(path, ignore_patterns, indent + "    ")


@task(name="tree")
def tree(ctx):
    """
    Print the project file structure, ignoring:
      - __pycache__
      - .pyc files
      - .pytest_cache

    Parameters
    ----------
    ctx : invoke.Context
        The invoke context.

    Returns
    -------
    None

    Bash:
        invoke tree
    """
    print_tree(".", IGNORED_PATTERNS)

# -----------------------------------------------------------------------------
# Build API Documentation Task
# -----------------------------------------------------------------------------


@task
def build_api(ctx):
    """
    Build Sphinx API documentation and copy it to the MkDocs docs folder.

    This task performs the following steps:
      1. Changes into the docs/ folder and builds the Sphinx documentation (HTML).
      2. Uses xcopy to copy the output from docs/_build/html to docs/api.

    Bash:
        invoke build-api
    """
    # Build the Sphinx documentation (assumes conf.py is in docs/)
    ctx.run("cd docs && python -m sphinx -b html . _build/html", pty=use_pty())
    # Copy the generated HTML files to the api/ folder within docs/
    ctx.run("xcopy /E /I /Y docs\\_build\\html docs\\api", pty=use_pty())


@task
def serve_mkdocs(ctx):
    """
    Serve the MkDocs site locally.

    This command starts a local development server for MkDocs,
    which automatically rebuilds the site when files change.

    Returns
    -------
    None

    Bash:
        invoke serve-mkdocs
    """
    ctx.run("mkdocs serve", pty=use_pty())

# -----------------------------------------------------------------------------
# Demo Sub-Collection
# -----------------------------------------------------------------------------


@task(name="examples")
def demo_examples(ctx):
    """
    Run the example runner demo.

    Parameters
    ----------
    ctx : invoke.Context
        The invoke context.

    Returns
    -------
    None

    Bash:
        invoke demo.examples
    """
    ctx.run("python -m demos.example_runner", pty=use_pty())


@task(name="options")
def demo_options(ctx):
    """
    Run the options demo.

    Parameters
    ----------
    ctx : invoke.Context
        The invoke context.

    Returns
    -------
    None

    Bash:
        invoke demo.options
    """
    ctx.run("python -m demos.containers.demo_options", pty=use_pty())


@task(name="market_env")
def demo_market_env(ctx):
    """
    Run the market_environment demo.

    Parameters
    ----------
    ctx : invoke.Context
        The invoke context.

    Returns
    -------
    None

    Bash:
        invoke demo.market-env
    """
    ctx.run("python -m demos.containers.demo_market_environment", pty=use_pty())


@task(name="stock")
def demo_stock(ctx):
    """
    Run the stock demo.

    Parameters
    ----------
    ctx : invoke.Context
        The invoke context.

    Returns
    -------
    None

    Bash:
        invoke demo.stock
    """
    ctx.run("python -m demos.containers.demo_stock", pty=use_pty())
# -----------------------------------------------------------------------------
# Create the Main Collection
# -----------------------------------------------------------------------------


ns = Collection(test, lint, format, stats, tree, docker, build_api, serve_mkdocs)

# Add Documentation sub-collection

# Add the demo sub-collection under the name "demo".
demo_ns = Collection(demo_examples, demo_options, demo_market_env, demo_stock)
ns.add_collection(demo_ns, name="demo")
