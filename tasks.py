"""
tasks.py

This module defines common development tasks using Invoke.
Run tasks with commands like:
    invoke test
    invoke lint
    invoke format
    invoke run
    invoke tree
    invoke stats
    invoke docker
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
    """
    return False if sys.platform == "win32" else True

@task(help={'verbose': "Run tests in verbose mode."})
def test(ctx, verbose=False):
    """
    Run the test suite using pytest.
    
    bash:
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
    
    bash:
        invoke lint
    """
    ctx.run("flake8", pty=use_pty())

@task
def format(ctx):
    """
    Format code using Black.
    
    bash:
        invoke format
    """
    ctx.run("black src tests examples", pty=use_pty())

@task(name="demo")
def demo(ctx):
    """
    Run the example runner (demo).

    bash:
        invoke demo
    """
    ctx.run("python -m demo.example_runner", pty=use_pty())

def print_tree(start_path, ignore_patterns, indent=""):
    """
    Recursively prints the directory tree, ignoring directories or files that match any of the ignore patterns.
    """
    try:
        items = sorted(os.listdir(start_path))
    except PermissionError:
        return
    for item in items:
        # Skip items that match any ignore pattern
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

    bash:
        invoke tree
    """
    print_tree(".", IGNORED_PATTERNS)


@task
def stats(ctx):
    """
    Run pygount to display a summary of code metrics (lines of code vs comments).

    bash:
        invoke stats
    """
    ctx.run("pygount -f summary .", pty=use_pty())
    
@task
def docker(ctx):
    """
    Build and run the Docker containers using docker-compose.
    
    This task runs:
        docker-compose up --build
    
    bash:
        invoke docker
    """
    ctx.run("docker-compose up --build", pty=use_pty())

# Create a collection of tasks
ns = Collection(test, lint, format, demo, stats, tree, docker)