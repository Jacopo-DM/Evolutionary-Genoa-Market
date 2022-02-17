#!/usr/bin/env python3

"""
Author:     Group 1
Date:       08.02.2022

This code is provided "As Is"
"""

# Default
import os
import sys
from pathlib import Path

# Thirdparty
import matplotlib.pyplot as plt


def em(to_bold: str) -> str:
    """Returns a string with with bold ANSI CODE decorators

    Args:
        to_bold: the string to which bolding should be applied
    """
    BOLD = "\033[;1m"
    CYAN = "\033[1;36m"
    RESET = "\033[0;0m"
    return f"{BOLD}{CYAN}{to_bold}{RESET}"


def log(to_print: str, *args) -> None:
    """Prints to standard output with nice formatting

    Args:
        to_print: A string to output to standard output
    """
    # Header Starter
    sys.stdout.write(em(" === "))

    # Make arguments bold
    em_args = [em(arg) for arg in args]

    # To Title
    to_print = to_print.title()

    # Print formatted line to output
    sys.stdout.write(to_print.format(*em_args))

    # Header Ender
    sys.stdout.write(em(" === "))

    # Newline
    sys.stdout.write("\n")


def save_figure(
    raw_plot_title: str, directory: str = "", overwrite: bool = True
) -> None:
    """
    Saves plt figure in local directory (current_directory/plots).
    Creates directory '/plots' if it does not exits.
    Can overwrite existing files, but needs permission before overwrite is executed

    Args:
        raw_asset_name: name of asset being plotted (likely title of plot)
        raw_plot_title: name of type of plot (likely title of subplot)
    """
    # Check existence of 'plots' folder
    dir_name = f"{os.getcwd()}/plots{directory}"
    Path(dir_name).mkdir(parents=True, exist_ok=True)

    # Generate file-name
    plot_title = raw_plot_title.lower().replace(" ", "_")
    file_name = f"{dir_name}/{plot_title}.png"

    # Save plot locally
    # log("Saving lot")
    if Path(file_name).is_file() and not overwrite:  # Check if file exists
        ow = input(f"Overwrite file {file_name}?[y/n]")
        if ow == "y":
            plt.savefig(file_name)
            log("Overwrite permitted, Save operation {}", "Succeeded")
        else:
            pass
            # log("Save operation {}", "Cancelled")
    else:
        plt.savefig(file_name)
        # log("Save operation {}", "Succeeded")
    plt.clf()
    plt.close()
