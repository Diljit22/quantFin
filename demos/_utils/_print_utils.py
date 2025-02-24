#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
_print_utils.py
===============
Utility functions for standardized printing with color and spacing.

This module provides functions to print titles, subtitles, general information,
and formatted descriptions with consistent styling using colorama.
"""

import textwrap
from colorama import Fore, Style, init

# Initialize colorama for Windows compatibility
init(autoreset=True)


def print_title(title: str) -> None:
    """
    Print a main title banner in a consistent color and style.

    Parameters
    ----------
    title : str
        The title text to be printed.

    Returns
    -------
    None
    """
    print()
    print(f"{Fore.CYAN}{'=' * 10} {title} {'=' * 10}{Style.RESET_ALL}")
    print()


def print_subtitle(subtitle: str) -> None:
    """
    Print a subtitle or section header in a consistent style.

    Parameters
    ----------
    subtitle : str
        The subtitle text to be printed.

    Returns
    -------
    None
    """
    print(f"{Fore.YELLOW}{'-' * 5} {subtitle} {'-' * 5}{Style.RESET_ALL}")


def print_info(info: str, indent: int = 2) -> None:
    """
    Print general informational text with a specified indent, wrapping to 80 columns.

    Parameters
    ----------
    info : str
        The information text to be printed.
    indent : int, optional
        The number of spaces to indent the text (default is 2).

    Returns
    -------
    None
    """
    wrapper = textwrap.TextWrapper(width=80, subsequent_indent=" " * indent)
    lines = wrapper.wrap(info)
    for line in lines:
        print(" " * indent + line)


def print_blank_line() -> None:
    """
    Print a blank line for spacing.

    Returns
    -------
    None
    """
    print()


def print_runner_title(title: str) -> None:
    """
    Print a main title/banner in a standardized runner style.

    Parameters
    ----------
    title : str
        The title text to be printed.

    Returns
    -------
    None
    """
    print("\n" + "=" * 80)
    print(f"{title.upper():^80}")
    print("=" * 80 + "\n")


def print_runner_subtitle(subtitle: str) -> None:
    """
    Print a subtitle or section header consistently for runner output.

    Parameters
    ----------
    subtitle : str
        The subtitle text to be printed.

    Returns
    -------
    None
    """
    print("-" * 80)
    print(f"{subtitle}")
    print("-" * 80)


def print_runner_description(description: str) -> None:
    """
    Print a formatted description for a section, wrapped to a comfortable width.

    Parameters
    ----------
    description : str
        The description text to be printed.

    Returns
    -------
    None
    """
    dedented_text = textwrap.dedent(description).strip()
    wrapped_text = textwrap.fill(dedented_text, width=78)
    print(wrapped_text)
    print()
