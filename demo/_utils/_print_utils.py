"""
_print_utils.py
===============

Utility functions for standardized printing with color and spacing.
"""

import textwrap
from colorama import Fore, Style, init

# Initialize colorama for Windows compatibility
init(autoreset=True)


def print_title(title: str) -> None:
    """
    Print a main title banner in a consistent color and style.
    """
    print()
    print(f"{Fore.CYAN}{'='*10} {title} {'='*10}{Style.RESET_ALL}")
    print()


def print_subtitle(subtitle: str) -> None:
    """
    Print a subtitle or section header in a consistent style.
    """
    print(f"{Fore.YELLOW}{'-'*5} {subtitle} {'-'*5}{Style.RESET_ALL}")


def print_info(info: str, indent: int = 2) -> None:
    """
    Print general info text with a given indent, wrapping the text to 80 columns.
    """
    wrapper = textwrap.TextWrapper(width=80, subsequent_indent=" " * indent)
    lines = wrapper.wrap(info)
    for line in lines:
        print(" " * indent + line)


def print_blank_line() -> None:
    """
    Print a blank line for spacing.
    """
    print()


def print_runner_title(title: str) -> None:
    """Print a main title/banner in a consistent style."""
    print("\n" + "=" * 80)
    print(f"{title.upper():^80}")
    print("=" * 80 + "\n")


def print_runner_subtitle(subtitle: str) -> None:
    """Print a subtitle or section header consistently."""
    print("-" * 80)
    print(f"{subtitle}")
    print("-" * 80)


def print_runner_description(description: str) -> None:
    """
    Print a textual description or introduction for a section,
    reflowed to a comfortable width.
    """
    dedented_text = textwrap.dedent(description).strip()
    wrapped_text = textwrap.fill(dedented_text, width=78)
    print(wrapped_text)
    print()
