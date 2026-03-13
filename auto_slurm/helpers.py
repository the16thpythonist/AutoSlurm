from __future__ import annotations

import time
import os
import re
import pathlib
import platform
import subprocess
import logging
import random
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from io import StringIO
from itertools import product
from typing import Iterator, Iterable, TypeVar, List

import rich_click
import jinja2 as j2
from rich.console import Console

T = TypeVar('T')

# The absolute path to the parent directory where this file is located in.
PATH: str = pathlib.Path(__file__).parent.resolve()

# This is the path to the templates directory which contains all the jinja2 template files
TEMPLATE_PATH: str = os.path.join(PATH, "templates")

# This is the template environment based on that template folder. This environment object can 
# be used to easily load the corresponding Template instances based on the files in that folder.
TEMPLATE_ENV = j2.Environment(
    loader=j2.FileSystemLoader(TEMPLATE_PATH),
    autoescape=j2.select_autoescape(),
)

# This can be used as a default logger wherever a logger can be supplied as a parameter/argument
# This logger will simply ignore all the logging calls but crucially it can be handled as any
# oher logger instance - reducing the need for if logger is not None: checks everywhere
NULL_LOGGER = logging.getLogger("auto_slurm")
NULL_LOGGER.addHandler(logging.NullHandler())


# ============================================================================
# Command Expansion Utilities
# ============================================================================
#
# These utilities enable parameter sweep syntax for batch job submission.
# Users can specify multiple parameter values in a single command, and these
# functions will expand them into multiple individual commands.
#
# Two syntaxes are supported:
#   1. Paired lists <[...]>  - Values across parameters are zipped together
#   2. Grid search  <{...}>  - Creates cartesian product of all values
#
# Example (paired):
#   "python train.py --lr=<[0.1, 0.01]> --bs=<[16, 32]>"
#   Expands to 2 commands: lr=0.1,bs=16 and lr=0.01,bs=32
#
# Example (grid):
#   "python train.py --lr=<{0.1, 0.01}> --bs=<{16, 32}>"
#   Expands to 4 commands: all combinations of lr and bs values
# ============================================================================


def split_top_level_commas(s: str) -> List[str]:
    """
    Split a string by commas, respecting nested bracket structures.

    This function splits a comma-separated string into parts, but only splits
    on commas that are at the "top level" - i.e., not enclosed within any
    brackets, braces, or parentheses. This is essential for parsing sweep
    syntax values that may themselves contain commas (e.g., nested lists).

    Algorithm:
        - Track bracket nesting depth as we scan the string
        - Only split on commas when depth is 0 (top level)
        - Raise an error if brackets are unbalanced

    Args:
        s (str): The string to split. May contain nested brackets, braces,
                 or parentheses that should be preserved.

    Returns:
        List[str]: List of comma-separated parts with leading/trailing
                   whitespace stripped from each part.

    Raises:
        ValueError: If brackets are unbalanced (more opens than closes
                   or vice versa).

    Examples:
        >>> split_top_level_commas("a, b, c")
        ['a', 'b', 'c']

        >>> split_top_level_commas("a, [b, c], d")
        ['a', '[b, c]', 'd']

        >>> split_top_level_commas("func(x, y), other(z)")
        ['func(x, y)', 'other(z)']

        >>> split_top_level_commas("[[1, 2], [3, 4]], 5")
        ['[[1, 2], [3, 4]]', '5']

    Note:
        All bracket types ([], {}, ()) are treated equivalently for
        nesting purposes. The function does not validate that bracket
        types match (e.g., "[}" would not raise an error for mismatched
        types, only for unbalanced counts).
    """
    parts = []
    current = []
    depth = 0

    for ch in s:
        # Track nesting depth for all bracket types
        if ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
            # Check for unbalanced closing bracket
            if depth < 0:
                raise ValueError("Unbalanced brackets")

        # Split on comma only at top level (depth == 0)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue

        current.append(ch)

    # Check for unbalanced opening brackets
    if depth != 0:
        raise ValueError("Unbalanced brackets")

    # Don't forget the last part (after the final comma, or the whole string)
    parts.append("".join(current).strip())
    return parts


def expand_commands(commands: List[str]) -> List[str]:
    """
    Expand commands containing parameter sweep syntax into multiple commands.

    This function processes a list of command strings and expands any that
    contain sweep syntax markers. Two types of sweep syntax are supported:

    1. **Paired Lists** ``<[value1, value2, ...]>``
       Multiple paired list markers in the same command are "zipped" together.
       All paired lists must have the same number of values.

       Example:
           Input:  ["python train.py --lr=<[0.1, 0.01]> --bs=<[16, 32]>"]
           Output: ["python train.py --lr=0.1 --bs=16",
                    "python train.py --lr=0.01 --bs=32"]

    2. **Grid Search** ``<{value1, value2, ...}>``
       Multiple grid markers create a cartesian product of all values.

       Example:
           Input:  ["python train.py --lr=<{0.1, 0.01}> --bs=<{16, 32}>"]
           Output: ["python train.py --lr=0.1 --bs=16",
                    "python train.py --lr=0.1 --bs=32",
                    "python train.py --lr=0.01 --bs=16",
                    "python train.py --lr=0.01 --bs=32"]

    Processing Order:
        Commands are processed in order, and expanded commands maintain
        relative ordering. A command without sweep syntax passes through
        unchanged.

    Args:
        commands (List[str]): List of command strings, possibly containing
                              sweep syntax markers.

    Returns:
        List[str]: Expanded list of commands with all sweep syntax resolved.
                   The length will be >= the input length (equal if no
                   expansion occurs).

    Raises:
        ValueError: If a single command mixes both <[]> and <{}> syntax.
        ValueError: If paired lists (<[]>) have different lengths.

    Examples:
        >>> # Paired expansion (zip behavior)
        >>> expand_commands(["echo <[a, b, c]>"])
        ['echo a', 'echo b', 'echo c']

        >>> # Grid expansion (cartesian product)
        >>> expand_commands(["echo <{1, 2}> <{x, y}>"])
        ['echo 1 x', 'echo 1 y', 'echo 2 x', 'echo 2 y']

        >>> # No expansion syntax - command passes through unchanged
        >>> expand_commands(["python script.py --arg=value"])
        ['python script.py --arg=value']

        >>> # Multiple commands, mixed expansion
        >>> expand_commands(["setup.py", "train.py --lr=<{0.1, 0.01}>"])
        ['setup.py', 'train.py --lr=0.1', 'train.py --lr=0.01']

    Note:
        - Values within sweep markers can contain nested brackets
        - Whitespace around values is stripped
        - Empty input list returns empty output list
    """
    expanded_commands = []

    for command in commands:
        # Find all instances of paired list <[...]> and grid search <{...}> syntax
        # Using non-greedy matching (.*?) to handle multiple markers per command
        bracket_matches = re.findall(r"<\[(.*?)\]>", command)
        brace_matches = re.findall(r"<\{(.*?)\}>", command)

        # Validate: cannot mix both syntaxes in the same command
        if bracket_matches and brace_matches:
            raise ValueError("Cannot mix <[]> and <{}> syntax in the same command.")

        if bracket_matches:
            # === Paired List Expansion (Zip Behavior) ===
            # Parse each <[...]> marker into a list of values
            options = [
                [subitem.strip() for subitem in split_top_level_commas(item)]
                for item in bracket_matches
            ]

            # Validate: all paired lists must have the same length
            if any(len(opt) != len(options[0]) for opt in options):
                raise ValueError("Paired lists must have the same length.")

            # Zip values together and create one command per tuple
            for values in zip(*options):
                temp_command = command
                # Replace each marker with its corresponding value
                # Using replace(..., 1) to handle duplicate patterns correctly
                for match, value in zip(bracket_matches, values):
                    temp_command = temp_command.replace(f"<[{match}]>", value, 1)
                expanded_commands.append(temp_command)

        elif brace_matches:
            # === Grid Search Expansion (Cartesian Product) ===
            # Parse each <{...}> marker into a list of values
            options = [
                [subitem.strip() for subitem in split_top_level_commas(item)]
                for item in brace_matches
            ]

            # Generate all combinations using cartesian product
            for values in product(*options):
                temp_command = command
                # Replace each marker with its corresponding value
                for match, value in zip(brace_matches, values):
                    # Note: need to escape braces in f-string: {{match}}
                    temp_command = temp_command.replace(f"<{{{match}}}>", value, 1)
                expanded_commands.append(temp_command)

        else:
            # === No Sweep Syntax ===
            # Command passes through unchanged
            expanded_commands.append(command)

    return expanded_commands


class Batched:
    """
    A generator wrapper that yields elements from an iterable in batches of a specified size.
    
    This class takes an iterable and a batch size and yields sublists containing elements
    from the original iterable. The last batch may contain fewer elements if the total
    number of elements is not evenly divisible by the batch size.
    
    The class also supports optional randomization of elements before batching, which
    creates a shuffled copy of the original iterable without mutating the original.
    
    Example:
        >>> numbers = list(range(10))
        >>> for batch in Batched(numbers, batch_size=3):
        ...     print(batch)
        [0, 1, 2]
        [3, 4, 5]
        [6, 7, 8]
        [9]
        
        >>> for batch in Batched(numbers, batch_size=3, randomize=True):
        ...     for element in batch:
        ...         print(element)  # Elements will be in random order
    """
    
    def __init__(self, iterable: Iterable[T], batch_size: int, randomize: bool = False):
        """
        Initialize the Batched generator wrapper.
        
        Args:
            iterable (Iterable[T]): The input iterable to be batched (e.g., list, tuple, generator).
            batch_size (int): The size of each batch. Must be greater than 0.
            randomize (bool, optional): If True, randomize the order of elements before batching.
                                      Defaults to False. Does not mutate the original iterable.
                                      
        Raises:
            ValueError: If batch_size is less than or equal to 0.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
            
        self.iterable = iterable
        self.batch_size = batch_size
        self.randomize = randomize
    
    def __iter__(self) -> Iterator[List[T]]:
        """
        Iterator method that yields batches of elements.
        
        Yields:
            List[T]: A list containing up to batch_size elements from the original iterable.
                    The last batch may contain fewer elements.
        """
        # Convert to list to allow for potential randomization and batching
        elements = list(self.iterable)
        
        # If randomize is True, create a shuffled copy without mutating the original
        if self.randomize:
            elements = elements.copy()
            random.shuffle(elements)
        
        # Yield batches of the specified size
        for i in range(0, len(elements), self.batch_size):
            yield elements[i:i + self.batch_size]


class RunTimer:
    def __init__(self, time_limit: int = 48):
        """Initialize the timer with a time limit in hours.

        Args:
            time_limit (int): The time limit in hours.
        """

        self.reset()
        self._time_limit = time_limit

    def reset(self):
        """Start the timer."""
        self._start_time = time.time()

    def time_limit_reached(self) -> bool:
        """Check if the time limit has been reached.

        Returns:
            bool: True if the time limit has been reached, False otherwise.
        """

        return (time.time() - self._start_time) > self._time_limit * 3600


def start_run(time_limit: int = 48):
    """Start the timer of a new run.

    Args:
        time_limit (int): The time limit in hours.

    Returns:
        RunTimer: The timer object.
    """

    run_timer = RunTimer(time_limit)
    return run_timer


def write_resume_file(command: str):
    """Write the command with which this process can be resumed to a file.

    Args:
        command (str): The command with which this process can be resumed.
    """

    slurm_id = os.environ.get("SLURM_JOB_ID", None)
    task_index = os.environ.get(
        "SLURM_SUBMIT_TASK_INDEX", 0
    )  # if this job has multiple tasks (processes) in one job

    if slurm_id is not None:
        os.makedirs("./.aslurm", exist_ok=True)
        with open(
            f"./.aslurm/{slurm_id}" + (f"_{task_index}") + ".resume",
            "w",
        ) as f:
            f.write(command)

    else:
        raise RuntimeError(
            "SLURM_JOB_ID not set, probably not running in a slurm environment."
        )


def get_version() -> str:
    """
    Returns the current version of the package as it is written in the ``VERSION`` file.
    
    Returns:
        str: The version of the package.
    """
    version_path: str = os.path.join(PATH, "VERSION")
    with open(version_path, "r") as file:
        version: str = file.read().strip()
        
    return version


def open_file_in_editor(file_path: str) -> None:
    """
    Opens the specified file in the system's default text editor.
    
    :param file_path: The path to the file to be opened.
    
    :returns: None
    """
    
    # Check the OS and open the file with the system default editor
    if platform.system() == 'Windows':
        os.startfile(file_path)  # Windows
    elif platform.system() == 'Darwin':
        subprocess.run(['open', file_path])  # macOS
    else:
        subprocess.run(['xdg-open', file_path])  # Linux


def create_slurm_jobs(
    fillers: dict,
    commands: list[str],
    options: dict[str, any],
    main_template: j2.Template = TEMPLATE_ENV.get_template('main.sh.j2'),
    resume_template: j2.Template = TEMPLATE_ENV.get_template('resume.sh.j2'),
    resume_script_name: str = "resume_0.sh",
) -> tuple[str, str]:
    """
    Generate SLURM job scripts for a batch of commands using Jinja2 templates.

    This function creates the main and resume SLURM job scripts by rendering the provided Jinja2 templates
    with the given commands, options, and additional context.

    Args:
        job_start_task_index (int):
            The starting index for the SLURM array job tasks. Used for job/task indexing in templates.
        fillers (dict):
            Additional variables to be passed to the templates for rendering.
        commands (list[str]):
            A list of shell commands to be executed by the SLURM job array. Each command typically represents a single task.
        options (dict[str, any]):
            Dictionary of SLURM/job options, including resource requirements and custom settings. May include 'gpus_per_task'.
        main_template (j2.Template, optional):
            Jinja2 template for the main SLURM job script. Defaults to the 'main.sh.j2' template.
        resume_template (j2.Template, optional):
            Jinja2 template for the resume SLURM job script. Defaults to the 'resume.sh.j2' template.

    Returns:
        tuple[str, str]:
            A tuple containing:
                - The rendered main SLURM job script as a string.
                - The rendered resume SLURM job script as a string.

    Notes:
        - The CUDA_VISIBLE_DEVICES assignments are generated based on the number of commands and GPUs per task.
        - The templates are expected to use the variables: fillers, commands, options, gpus, and gpus_per_task.
        - This function does not write any files; it only returns the rendered script contents.
    """

    # This list will be the same length as the commands list and contain the strings which will be used 
    # for the CUDA_VISIBLE_DEVICES environment variable in the slurm job script based on the number
    # of gpus_per_task configured in the options.
    gpus: list[str] = []
    gpus_per_task = options.get("gpus_per_task", None)
    if gpus_per_task is not None and gpus_per_task > 0:
        for i in range(0, len(commands) * gpus_per_task, gpus_per_task):
            gpus.append(",".join(str(j) for j in range(i, i + gpus_per_task)))

    main_script_content: str = main_template.render(
        fillers=fillers,
        commands=commands,
        options=options,
        gpus=gpus,
        resume_script_name=resume_script_name,
    )

    resume_script_content: str = resume_template.render(
        fillers=fillers,
        commands=commands,
        options=options,
        gpus_per_task=gpus_per_task,
        gpus=gpus,
        resume_script_name=resume_script_name,
    )
    
    return main_script_content, resume_script_content


@contextmanager
def suppress_console_output():
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        yield