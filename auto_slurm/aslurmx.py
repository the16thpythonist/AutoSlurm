import os
import sys
import pathlib
import datetime
import subprocess
import uuid
import logging
import math
import rich_click as click
import io
from contextlib import redirect_stdout

import rich
import yaml
import hydra
import omegaconf
from tqdm import tqdm
from more_itertools import chunked
from rich.pretty import pprint
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.style import Style
from rich.syntax import Syntax
from rich.columns import Columns
from rich.padding import Padding
from auto_slurm.helpers import PATH, TEMPLATE_PATH, TEMPLATE_ENV
from auto_slurm.helpers import NULL_LOGGER
from auto_slurm.helpers import Batched
from auto_slurm.helpers import open_file_in_editor
from auto_slurm.helpers import get_version
from auto_slurm.helpers import create_slurm_jobs
from auto_slurm.config import AutoSlurmConfig
from auto_slurm.config import GeneralConfig, Config
from jinja2 import FileSystemLoader, ChoiceLoader

class RichLogo:
    """
    A rich display which will show the ASlurmX logo in ASCII art when printed.
    """

    STYLE = Style(bold=True, color="white")

    def __rich_console__(self, console, options):
        text_path = os.path.join(TEMPLATE_PATH, "logo_text.txt")
        with open(text_path) as file:
            text_string: str = file.read()
            text = Text(text_string, style=self.STYLE)
            
        image_path = os.path.join(TEMPLATE_PATH, "logo_image.txt")
        with open(image_path) as file:
            image_string: str = file.read()
            # Replace \e with actual escape character and create Text from ANSI
            ansi_string = image_string.replace('\\e', '\033')
            image = Text.from_ansi(ansi_string)
            
        side_by_side = Columns([image, text], equal=True, padding=(0, 3))
        yield Padding(side_by_side, (1, 3, 1, 3))


class RichHelp:
    """
    A rich display which will show the "help" section for the ASlurmX CLI tool when printed.
    This help sections contains special formatting for the various example commands.
    """
    
    def __rich_console__(self, console, options):
        yield "AutoSlurm Command Line Interface X.\n"
        yield Text((
            "This is a tool to simplify the scheduling of SLURM jobs across different HPC environments. "
            "You can schedule a new slurm job using the 'cmd' command like this:"
        ))
        yield Padding(Syntax((
            "aslurmx -cn <config_name> cmd python script.py --arg1=100"
        ), lexer='bash', theme='monokai', line_numbers=False), (1, 5))
        yield Text((
            "Everything after the 'cmd' keyworkd will be used as the actual command to be scheduled in SLURM.\n"
        ))
        yield Text((
            "For instance, the following command will schedule the script 'train.py' to be executed on the "
            "BWUni cluster 3.0, using 20G of memory and by activating the conda environment 'myenv'"
        ))
        yield Padding(Syntax((
            "aslurmx -cn bwuni_1gpu_a100 -o conda_env=myenv,mem=20G cmd python train.py"
        ), lexer='bash', theme='monokai', line_numbers=False), (1, 5))
        yield Text((
            "You can even schedule multiple commands at the same time by using multiple 'cmd' commands and "
            "also control how the available GPUs "
            "should be distributed across those tasks. The following command will schedule two jobs, allocating "
            "2 GPUs of a 4-GPU node to each task. The tasks will be executed in parallel, but still "
            "be bundled in the same job:"  
        ))
        yield Padding(Syntax((
            "aslurmx ---config=bwuni_4gpu_h100 --gpus-per-task=2 cmd python train.py cmd python train.py"
        ), lexer='bash', theme='monokai', line_numbers=False), (1, 5))

class RichConfigList:
    
    def __init__(self, config_map: dict[str, dict]):
        self.config_map = config_map
        
        self.column_names: list[str] = [
            'Config Name',
            'Partition',
            'Time',
            'Memory',
            'CPUs',
            'GRES',
        ]
        
        self.rows: list[list] = []
        for config_name, config_content in self.config_map.items():
            if 'default_fillers' in config_content:
                row: list = [
                    config_name,
                    config_content['default_fillers'].get('partition', 'N/A'),
                    config_content['default_fillers'].get('time', 'N/A'),
                    config_content['default_fillers'].get('mem', 'N/A'),
                    config_content['default_fillers'].get('cpus', 'N/A'),
                    config_content['default_fillers'].get('gres', 'N/A'),
                ]
                self.rows.append(row)
        
        # Sort rows alphabetically by config name (first column)
        self.rows.sort(key=lambda row: row[0])
        
    def __rich_console__(self, console, options):
        
        table = Table(
            show_header=True,
            header_style="bold magenta",
            expand=True,
            title="Available AutoSlurm Configs"
        )
        # Make the first column expand to fill available space
        table.add_column(self.column_names[0], style="bold", no_wrap=False, ratio=2)
        for col in self.column_names[1:]:
            table.add_column(col)

        for row in self.rows:
            # Make the first column bold
            table.add_row(f"[bold]{row[0]}[/bold]", *[str(cell) for cell in row[1:]])

        yield table


class KeyValueList(click.ParamType):
    """
    A custom Click parameter type for parsing comma-separated key-value pairs from the command line.

    This class allows users to specify multiple key-value pairs as a single command-line argument,
    using the format: key1=value1,key2=value2,... The resulting value is converted into a Python
    dictionary mapping each key to its corresponding value.

    Example usage in a Click option:
        @click.option('--overwrite-fillers', '-o', type=KeyValueList(), default={}, help='Overwrite fillers for the config.')

    Example command-line input:
        --overwrite-fillers time=01:00:00,mem=16G,cpus=4

    This would result in:
        {'time': '01:00:00', 'mem': '16G', 'cpus': '4'}

    Notes:
        - Whitespace around keys and values is stripped.
        - If the input is empty or None, an empty dictionary is returned.
        - Invalid pairs (missing '=') or empty keys will raise a Click error.
    """

    name = "keyvaluelist"

    def convert(self, value, param, ctx):
        """
        Converts a comma-separated string of key-value pairs into a dictionary.

        Args:
            value (str): The input string from the command line, e.g., "key1=val1,key2=val2".
            param: The Click parameter object (unused).
            ctx: The Click context object (unused).

        Returns:
            dict: A dictionary mapping keys to values as parsed from the input string.

        Raises:
            click.BadParameter: If a pair does not contain '=', or if a key is empty.

        Example:
            >>> KeyValueList().convert("foo=bar,baz=qux", None, None)
            {'foo': 'bar', 'baz': 'qux'}
        """
        # If the input is empty or None, return an empty dictionary
        if not value:
            return {}

        result = {}
        # Split the input string by commas to get individual key-value pairs
        pairs = value.split(",")
        for pair in pairs:
            # Each pair must contain an '=' character
            if "=" not in pair:
                self.fail(
                    f"Invalid key-value pair: '{pair}'. Expected format: key=value",
                    param,
                    ctx
                )
            # Split only on the first '=' to allow '=' in values
            key, val = pair.split("=", 1)
            key = key.strip()
            val = val.strip()
            # Key must not be empty
            if not key:
                self.fail(
                    f"Empty key in pair: '{pair}'",
                    param,
                    ctx
                )
            result[key] = val
        return result


class ASlurm(click.RichGroup):
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.rich_logo = RichLogo()
        self.rich_help = RichHelp()
        
        ## --- attribute setup ---
        
        # This dict will store the global options that are passed to the aslurm base command 
        # such as the name of the config to use etc.
        self.options: dict[str, any] = {}
        
        ## --- registering commands ---
        # The individual commands are registered
        self.add_command(self.cmd_command)
        
        self.config_group.add_command(self.list_configs_command)
        self.config_group.add_command(self.edit_configs_command)
        self.config_group.add_command(self.where_configs_command)
        self.add_command(self.config_group)
        
        ## --- initialization ---
        # The following section of the constructor performs common initialization tasks which 
        # will be required for all the commands.
        
        #  ~ loading general config
        # This will load the general configuration file that is shipped with the package
        self.aslurm_config: AutoSlurmConfig = AutoSlurmConfig()
        self.aslurm_config.setup_if_necessary()
        
        general_config_path: str = os.path.relpath(
            path=self.aslurm_config.folder_path, 
            start=PATH
        )
        with hydra.initialize(general_config_path, version_base=None):
            cfg = hydra.compose(config_name='general_config')
            cfg_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            self.general_config: GeneralConfig = GeneralConfig(**cfg_dict)
                    
        # ~ config discovery
        # Here we want to discover all the possible configuration files that are available on the current 
        # system.
        
        # This list will contain the absolute string paths to all of the *folders* containing 
        # viable config files which may be used by the auto-slurm system.
        self.config_source_paths: list[str] = [
            # We put the custom folder first here such that we can use it to override the default configs 
            # that are shipped with the package if we want to.
            self.aslurm_config.configs_folder_path,
            # This is the configs folder that is shipped with the package.
            os.path.join(PATH, 'configs')
        ]
        
        ## --- template environment update ---
        
        # Add a custom templates folder (e.g., ~/.aslurm/templates) as the highest-priority source
        custom_templates_folder = self.aslurm_config.templates_folder_path
        if os.path.isdir(custom_templates_folder):
            # Prepend the custom loader so its templates override the defaults
            TEMPLATE_ENV.loader = ChoiceLoader([
                FileSystemLoader(custom_templates_folder),
                TEMPLATE_ENV.loader
            ])
    
    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """
        This method overrides the default "format_help" function of the click.Group class.
        This method is used to override the help string that is printed for the --help 
        option of the overall group.
        """
        # Before printing the help text we want to print the logo
        rich.print(self.rich_logo)
        
        rich.print(self.rich_help)
        
        self.format_usage(ctx, formatter)
        self.format_options(ctx, formatter)
        self.format_epilog(ctx, formatter)
    
    # == "config" commands ==
    # Commands to interact with the configuration files.
    
    @click.group('config', short_help='Various commands to interact with the configuration files.')
    @click.pass_obj
    def config_group(self):
        """
        Exposes commands to interacti with the individual config files that define the various cluster 
        environments and their default settings.
        """
        pass
    
    @click.command('list', short_help='List all the configuration files that are available.')
    @click.pass_obj
    def list_configs_command(self):
        """
        Outputs a list of all the available configurations
        """
        ## --- config file discovery ---
        # This data structure will store the mapping between the name of a config file 
        # and the absolute string path to the file itself.
        config_name_path_dict: dict[str, str] = {}

        for source_path in self.config_source_paths:
            
            try:
                for file_name in os.listdir(source_path):
                    
                    if file_name.endswith('.yaml') or file_name.endswith('.yml'):
                        # We want to use the name of the config file without the extension as the key.
                        config_name = file_name.rsplit('.', 1)[0]
                        # We store the absolute path to the config file as the value.
                        config_name_path_dict[config_name] = os.path.join(source_path, file_name)
            except (FileNotFoundError, NotADirectoryError, OSError):
                continue

        ## --- config file loading ---
        # In the next step we go through the previous mapping and actually load the content of these 
        # config files into another dictionary.
        
        config_name_content_dict: dict[str, dict] = {}
        for config_name, config_path in config_name_path_dict.items():
            try:
                with open(config_path, 'r') as file:
                    config_content = yaml.safe_load(file)
                config_name_content_dict[config_name] = config_content
            except Exception as e:
                click.echo(f"Failed to load {config_path}: {e}", err=True)
            
        ## --- Display the table ---
        print()
        rich_config_list = RichConfigList(config_map=config_name_content_dict)
        rich.print(rich_config_list)
        
    @click.command('edit', short_help='Edit a configuration file.')
    @click.argument('config_name', type=str)
    @click.pass_obj
    def edit_configs_command(self, config_name: str):
        """
        Open the config file identified by the given CONFIG_NAME in the default text editor.
        """
        
        ## --- config file discovery ---
        # This data structure will store the mapping between the name of a config file 
        # and the absolute string path to the file itself.
        config_name_path_dict: dict[str, str] = {}

        for source_path in self.config_source_paths:
            
            try:
                for file_name in os.listdir(source_path):
                    
                    if file_name.endswith('.yaml') or file_name.endswith('.yml'):
                        # We want to use the name of the config file without the extension as the key.
                        config_name_ = file_name.rsplit('.', 1)[0]
                        # We store the absolute path to the config file as the value.
                        config_name_path_dict[config_name_] = os.path.join(source_path, file_name)
            except (FileNotFoundError, NotADirectoryError, OSError):
                continue
            
        ## --- edit config file ---
        # We can retrieve the path oto the config file itself using the data structure that we defined above
        # and then use the function to open that file in a text editor.
        
        if config_name not in config_name_path_dict:
            click.secho(f'⚠️ Config file "{config_name}" not found in any of the config source paths.', fg='red')
            sys.exit(1)
            
        config_path: str = config_name_path_dict[config_name]
        # This function will open the given config file in the default text editor.
        open_file_in_editor(config_path)
    
    @click.command('where', short_help='Show the location where config files are stored.')
    @click.pass_obj
    def where_configs_command(self):
        """
        Show the location where all the config files are being stored in.
        """
        # Create a rich table to display the config paths
        table = Table(
            show_header=True,
            header_style="bold magenta",
            expand=True,
            title="Config File Storage Locations"
        )
        table.add_column("Priority", style="bold cyan", width=8)
        table.add_column("Path", style="dim")
        table.add_column("Status", justify="center", width=10)
        
        for i, path in enumerate(self.config_source_paths, 1):
            # Check if the path exists and add appropriate status
            if os.path.exists(path):
                status = "[green]✓ Exists[/green]"
            else:
                status = "[yellow]⚠ Missing[/yellow]"
            
            # Show priority (1 = highest priority)
            priority = str(i)
            if i == 1:
                priority = f"[bold]{priority}[/bold] (highest)"
            
            table.add_row(priority, path, status)
        
        rich.print()
        rich.print(table)
        
        # Add helpful explanation
        text = Text("\nNote: Config files in higher priority locations override those in lower priority locations.", 
                   style="dim italic")
        rich.print(Padding(text, (0, 2)))
    
    # == "cmd" commands ==
    # Commands to actually pass custom things to be scheduled in slurm.
    
    @click.command('cmd',
                   context_settings=dict(ignore_unknown_options=True, allow_extra_args=True), 
                   short_help='Add custom commands to be scheduled in SLURM.')
    @click.argument('args', nargs=-1, type=click.UNPROCESSED)
    @click.pass_obj
    def cmd_command(self, args):
        """
        Add custom commands to be scheduled in SLURM.
        """
        
        # Combine known and unknown args
        args_raw = ['cmd', *args]
        print(args_raw)
        
        # 1) find the individual commands
        # The "cmd" command may actually be composed of multiple "cmd" commands at the same time 
        # and in this first step we want to find all the ones for the given invocation.
        
        # This data structure will store all the individual commands that are found in the 
        # given command line arguments as individual strings.
        commands: list[str] = self.extract_commands_from_args(args_raw)
        click.echo(f'preparing to submit {len(commands)} commands...')
        
        # 2) config loading
        # Here we want to load the config that is specified by the user via the `-cn` option.
        # We search in all of the config source paths for a file with the given name.

        # This method will load the config object based on the given config name from one of the 
        # available config source paths.
        config: Config = self.load_config(config_name=self.options['config_name'])
        click.echo(f'✅ loaded config: {self.options["config_name"]}')
                
        # 3) script folder
        # Next, we need to create the folder where the temporary slurm scripts will actually 
        # be stored in.
        
        # This method will make sure to create a new folder relative to the current working directory 
        # in which we can store the slurm scripts that are generated.
        scripts_path: str = self.create_scipts_folder(self.options['archive_path'])
        click.echo(f'✅ created scripts folder @ {scripts_path}')
        
        # 4) job splitting
        
        # We use the default max_tasks value from the config unless it it superseded by a command line 
        # option.
        max_tasks: int | None = config.default_fillers.get('max_tasks')
        max_tasks = self.options['max_tasks'] if self.options['max_tasks'] is not None else max_tasks
        
        # With this section we implement the command splitting logic. Either we put all the commands into a 
        # single job / slurm script (if `--same` is set) or we split them into multiple jobs.
        # In this data structure we will store the individual commands lists that will result in individual 
        # slurm scripts. 
        commands_list: list[list[str]] = []
        if self.options['same']:
            commands_list.append(commands)
        elif max_tasks is not None and max_tasks > 0:
            commands_list = list(chunked(commands, max_tasks))
        else:
            commands_list.extend([[command] for command in commands])
            
        # 5) script generation
        # In this step we will generate the actual slurm scripts based on the commands that we have
        # collected in the previous step.
        
        for job_index, _commands in enumerate(commands_list):
            
            # --- assembling the fillers ---
            # The filler values that we'll use in the templates to assemble the scripts are 
            # a combination of the global filler values as defaults which are then overwritten 
            # by the values that are specified in the specific config file.
            fillers: dict[str, any] = self.general_config.global_fillers
            
            # --- default venv discorvery ---
            # As a small convenience feature we want to automatically detect if there is a venv file in 
            # the current working directory of where the command was invoked and then add that as 
            # the default filler if possible.
            # We specifically do this as the first step because we definitely want the user 
            # supplied value to have higher priority and replace this default.
            venv_path: str | None = self.discover_venv(os.getcwd())
            if venv_path:
                fillers['venv'] = venv_path
            
            # The fillers from the global defaults and the command line.
            fillers.update(config.default_fillers)
            fillers.update(self.options['overwrite_fillers'])
                        
            # --- creating SLURM scripts ---
            # Here we actually create the slurm scripts using the helper function. This helper function 
            # will fill the jinja templates with the content based on the fillers and the commands.
            main_content, resume_content = create_slurm_jobs(
                commands=_commands,
                fillers=fillers,
                options=self.options,
                main_template=TEMPLATE_ENV.get_template('main.sh.j2'),
                resume_template=TEMPLATE_ENV.get_template('resume.sh.j2'),
            )
            
            # Now we actually write the files to the scripts folder.
            main_path = os.path.join(scripts_path, f'main_{job_index}.sh')
            with open(main_path, 'w') as main_file:
                main_file.write(main_content)
                
            resume_path = os.path.join(scripts_path, f'resume_{job_index}.sh')
            with open(resume_path, 'w') as resume_file:
                resume_file.write(resume_content)
                
            # After writing the files - only if this is NOT a dry run - we use subprocess to actually start 
            # the job using sbatch.
            if not self.options['dry_run']:
                
                try:
                    sbatch_command = ['sbatch', main_path]
                    result = subprocess.run(sbatch_command, capture_output=True, text=True, check=True)

                    output = result.stdout
                    if "Submitted" in output:
                        # Extract the job ID from the output:
                        # The job ID is included in the output as such: "Submitted batch job 123456"
                        job_id: str = output.strip().split()[-1]
                        click.echo(f'🚀 Submitted job {job_index} with SLURM ID {job_id}! - {_commands[0]}...')
                    else:
                        raise ValueError("Unexpected sbatch output format: " + output)
                    
                except (subprocess.CalledProcessError, ValueError) as e:
                    click.echo(f'⚠️ Failed to submit job {job_index}!', err=True)
                    print(e)
                    sys.exit(1)
        
        click.echo()
        click.echo('You may check on the status of your jobs using the `squeue` command.')
        
    # == Helper methods ==
    # The following methods do not implement any commands but rather provide utility functions
    # that are used by the commands above.
        
    def discover_venv(self, path: str) -> str | None:
        """
        Discover and return the path to a virtual environment folder within the specified directory.
        
        This method searches for common virtual environment directory names within the given path
        and returns the absolute path to the first virtual environment found. It looks for standard
        virtual environment folder names including 'venv', '.venv', 'env', '.env', and 'virtualenv'.
        The method prioritizes hidden directories (starting with '.') as they are commonly used
        for virtual environments to keep them out of regular file listings.
        
        Args:
            path (str): The directory path to search for virtual environment folders.
                       Should be an absolute or relative path to an existing directory.
                       
        Returns:
            str | None: The absolute path to the discovered virtual environment directory,
                       or None if no virtual environment is found in the specified path.
                       
        Example:
            >>> # Search for venv in current working directory
            >>> venv_path = self.discover_venv('/home/user/project')
            >>> print(venv_path)
            '/home/user/project/.venv'
            
            >>> # No venv found
            >>> venv_path = self.discover_venv('/home/user/empty_dir')
            >>> print(venv_path)
            None
            
        Notes:
            - The method returns the first virtual environment found based on priority order
            - Priority order: .venv, venv, .env, env, virtualenv
            - Only returns directories that actually exist and are accessible
            - Does not validate that the discovered directory is a valid Python virtual environment
            - Case-sensitive search (looks for exact name matches)
            
        Raises:
            OSError: If the specified path does not exist or is not accessible
            
        See Also:
            create_scipts_folder: Creates directories for SLURM script storage
            load_config: Loads configuration files from multiple source paths
        """
        # Define common virtual environment directory names in priority order
        # Hidden directories (starting with '.') are prioritized as they're commonly used
        venv_names = ['.venv', 'venv', '.env', 'env', 'virtualenv']
        
        try:
            # Check if the provided path exists and is a directory
            if not os.path.isdir(path):
                return None
                
            # Search for virtual environment directories in the specified path
            for venv_name in venv_names:
                venv_path = os.path.join(path, venv_name)
                
                # Return the absolute path if a virtual environment directory is found
                if os.path.isdir(venv_path):
                    return os.path.abspath(venv_path)
                    
        except (OSError, PermissionError):
            # Return None if path is not accessible
            return None
            
        # No virtual environment found
        return None
        
    def extract_commands_from_args(self, args: list[str]) -> list[str]:
        """
        Extracts individual command strings from a list of arguments, where each command is prefixed by the keyword "cmd".
        This method scans through the provided list of arguments (`args`), searching for occurrences of the string "cmd".
        For each "cmd" found, it collects all subsequent arguments up to the next "cmd" or the end of the list, and joins them
        into a single command string separated by spaces. Each such command string is added to the returned list.
        
        Args:
            args (list[str]): A list of strings representing arguments, where each command is introduced by the keyword "cmd".
                              For example: ["cmd", "echo", "hello", "cmd", "ls", "-l"]
        Returns:
            list[str]: A list of command strings, each assembled from the arguments following a "cmd" keyword up to the next "cmd"
                       or the end of the list. Empty commands (i.e., "cmd" not followed by any arguments) are ignored.
        Example:
            >>> extract_commands_from_args(["cmd", "echo", "hello", "cmd", "ls", "-l"])
            ['echo hello', 'ls -l']
        Notes:
            - If "cmd" appears consecutively (e.g., ["cmd", "cmd", "ls"]), empty commands are ignored.
            - Arguments before the first "cmd" are ignored.
            - The method does not validate the content of the commands, only their extraction based on the "cmd" delimiter.
        """
        # In this list we will store all the individual assembled commands that are found 
        # in the argument list
        commands: list[str] = []
        
        # We iterate through the arguments, as soon as we find a "cmd" argument we start collecting
        # all the following arguments until we find the next "cmd" argument or reach the end of the list.
        # We combine all the arguments in between into a single command string by inserting whitespaces.
        i = 0
        while i < len(args):
            if args[i] == "cmd":
                j = i + 1
                while j < len(args) and args[j] != "cmd":
                    j += 1
                command = " ".join(args[i+1:j])
                if command.strip():
                    commands.append(command)
                i = j
            else:
                i += 1
                
        return commands

    def load_config(self, config_name: str) -> Config:
        """
        Loads a configuration file with the specified name from a list of possible source directories.
        This method iterates over all directories specified in `self.config_source_paths`, attempting to load
        a configuration file matching `config_name` from each location using Hydra. The first successfully
        loaded configuration is returned as an instance of the `Config` class. If no configuration file with
        the given name is found in any of the source paths, a `FileNotFoundError` is raised.
        
        Args:
            config_name (str): The name of the configuration file to load (without file extension).
        Returns:
            Config: An instance of the `Config` class populated with the loaded configuration data.
        Raises:
            FileNotFoundError: If no configuration file with the specified name is found in any of the
                configured source directories.
        Notes:
            - Only the first successfully loaded configuration is used; subsequent paths are not checked.
            - If a configuration is missing in a particular source path, the method silently continues to the
              next path.
        """
        
        # ~ config loading
        # We'll iterate over all of the config source paths and try to load the config with the given 
        # name from each of them until we find one that works.
        config: Config | None = None
        for source_path in self.config_source_paths:
            
            try:
                with hydra.initialize_config_dir(source_path, version_base=None):
                    cfg = hydra.compose(config_name=self.options['config_name'])
                    cfg_dict = omegaconf.OmegaConf.to_container(
                        cfg, resolve=True, throw_on_missing=True
                    )
                    config: Config = Config(**cfg_dict)
                    break
                
            except hydra.errors.MissingConfigException as exc:
                continue

        # If "config" remains None after the loop, that means that no config with the given name was found 
        # in any of the possible locations...
        if config is None:
            raise FileNotFoundError(
                f'There exists no AutoSlurm config file with the name "{config_name}"!. '
                f'Please check the list of available configs...'
            )
            
        return config

    def create_scipts_folder(self, path: str | None = None) -> str:
        """
        Creates a uniquely named scripts folder within a hidden '.aslurm' directory in the current working directory.
        The folder name is generated using the current date and time (formatted as 'YYYY-MM-DD_HH-MM-SS') 
        concatenated with the first 7 characters of a newly generated UUID, ensuring uniqueness for each invocation.
        
        Args:
            path (str): The base path where the '.aslurm' directory should be created.
        Returns:
            str: The absolute path to the newly created scripts folder.
        Side Effects:
            - Creates the '.aslurm' directory in the current working directory if it does not already exist.
            - Creates a new subdirectory within '.aslurm' with a unique name.
        Example:
            >>> folder_path = self.create_scipts_folder()
            >>> print(folder_path)
            '/current/working/dir/.aslurm/2024-06-10_15-30-45_1a2b3c4'
        Notes:
            - If the method is called multiple times in quick succession, the UUID ensures that folder names do not collide.
            - The method uses the current working directory as the base path for folder creation.
        """
        
        if path is None:
            path = os.getcwd()
        
        scripts_folder_path: str = os.path.join(
            path,
            '.aslurm',
            f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{str(uuid.uuid4())[0:7]}'
        )
        os.makedirs(scripts_folder_path, exist_ok=True)
        return scripts_folder_path




class ASlurmSubmitter:
    """
    A programmatic interface for submitting batches of commands to SLURM using AutoSlurm configurations.
    
    The ASlurmSubmitter class provides a high-level Python API for scheduling and managing SLURM jobs
    without directly interacting with the command-line interface. It's particularly well-suited for
    automating the submission of large numbers of commands or integrating SLURM job submission into
    Python workflows and scripts.
    
    Key Features:
    - **Batch Processing**: Groups multiple commands into SLURM jobs based on configurable batch sizes
    - **GPU Assignment**: Automatically assign different GPUs to commands within a batch using CUDA_VISIBLE_DEVICES
    - **Command Queuing**: Add commands incrementally before submitting them all at once
    - **Configuration Management**: Uses AutoSlurm's configuration system for consistent job settings
    - **Randomization Support**: Optional command randomization for better resource utilization
    - **Dry Run Mode**: Test job submission without actually scheduling SLURM jobs
    - **Flexible Archiving**: Configure where SLURM scripts are stored
    
    Typical Workflow:
    1. Create an ASlurmSubmitter instance with desired configuration
    2. Add commands to the queue using add_command()
    3. Submit all queued commands using submit()
    
    Example:
        >>> # Basic usage: Create a submitter for batch processing
        >>> submitter = ASlurmSubmitter(
        ...     config_name='gpu_cluster',
        ...     batch_size=4,
        ...     randomize=True
        ... )
        >>>
        >>> # Add multiple training commands
        >>> for lr in [0.001, 0.01, 0.1]:
        ...     submitter.add_command(f'python train.py --learning_rate={lr}')
        >>>
        >>> # Submit all commands (will create jobs with 4 commands each)
        >>> submitter.submit()

        >>> # GPU assignment: Distribute commands across GPUs
        >>> submitter = ASlurmSubmitter(
        ...     config_name='haicore_4gpu',
        ...     batch_size=4,
        ...     gpus_per_task=1  # Assign 1 GPU per command
        ... )
        >>>
        >>> # Add 4 commands - each will get its own GPU (0, 1, 2, 3)
        >>> for i in range(4):
        ...     submitter.add_command(f'python train.py --model=model{i}')
        >>>
        >>> submitter.submit()
        >>> # Result: All 4 commands run in parallel, each on a different GPU
        >>> # Command 0: CUDA_VISIBLE_DEVICES=0
        >>> # Command 1: CUDA_VISIBLE_DEVICES=1
        >>> # Command 2: CUDA_VISIBLE_DEVICES=2
        >>> # Command 3: CUDA_VISIBLE_DEVICES=3
    
    Batch Size Considerations:
    - **batch_size=1**: Each command gets its own SLURM job (maximum parallelism)
    - **batch_size>1**: Multiple commands are grouped into single jobs (resource efficiency)
    - **Large batch_size**: Fewer jobs but longer execution times per job
    
    Configuration Integration:
    The submitter leverages AutoSlurm's configuration system, inheriting settings like:
    - Resource allocation (CPUs, memory, GPUs)
    - Time limits and partitions
    - Module loading and environment setup
    - Custom templates and job scripts
    
    Thread Safety:
    This class is not thread-safe. If using in multi-threaded environments,
    external synchronization is required when adding commands or submitting jobs.
    
    Attributes:
        config_name (str): Name of the AutoSlurm configuration to use
        batch_size (int): Number of commands to group into each SLURM job
        randomize (bool): Whether to randomize command order before batching
        gpus_per_task (int | None): Number of GPUs to assign per command (None = no GPU assignment)
        num_gpus (int | None): Total number of GPUs available for distribution
        logger (logging.Logger): Logger instance for operation tracking
        commands (list[str]): Internal queue of commands awaiting submission
        options (dict): Internal options dictionary for CLI interface
        cli (ASlurm): Internal CLI interface instance
        ctx (click.Context): Click context for command execution
    
    See Also:
        - ASlurm: The underlying command-line interface
        - AutoSlurmConfig: Configuration management system
        - Batched: Helper class for command batching with randomization
    """
    def __init__(self,
                 config_name: str,
                 batch_size: int = 1,
                 randomize: bool = False,
                 logger: logging.Logger = NULL_LOGGER,
                 dry_run: bool = False,
                 overwrite_fillers: dict[str, str] = {},
                 archive_path: str = os.getcwd(),
                 parallel: bool = False,
                 gpus_per_task: int | None = None,
                 num_gpus: int | None = None,
                 ):
        """
        Initialize an ASlurmSubmitter instance with the specified configuration and options.
        
        This constructor sets up all necessary components for SLURM job submission, including
        loading the AutoSlurm configuration, initializing the CLI interface, and preparing
        the internal command queue.
        
        Args:
            config_name (str): Name of the AutoSlurm configuration file to use (without extension).
                              This must correspond to a YAML file in one of the AutoSlurm config directories.
                              Examples: 'gpu_cluster', 'cpu_partition', 'high_memory'
                              
            batch_size (int, optional): Number of commands to group into each SLURM job. 
                                       Defaults to 1 (each command gets its own job).
                                       - Values > 1 create fewer jobs but longer execution times
                                       - Consider resource limits and job queue policies when choosing
                                       
            randomize (bool, optional): Whether to randomize the order of commands before batching.
                                       Defaults to False. When True:
                                       - Helps distribute computational load more evenly
                                       - Useful for parameter sweeps or similar workloads
                                       - Commands within each batch are still executed sequentially
                                       
            logger (logging.Logger, optional): Logger instance for tracking operations and debugging.
                                              Defaults to NULL_LOGGER (no logging).
                                              Recommended for production use to track job submissions.
                                              
            dry_run (bool, optional): If True, generates SLURM scripts but does not submit them.
                                     Defaults to False. Useful for:
                                     - Testing configuration and script generation
                                     - Validating command syntax
                                     - Debugging job submission issues
                                     
            archive_path (str, optional): Directory where generated SLURM scripts will be stored.
                                         Defaults to current working directory.
                                         - Scripts are organized in timestamped subdirectories
                                         - Should be accessible to the SLURM cluster
                                         - Consider disk space for large numbers of jobs

            parallel (bool, optional): Whether to run commands in parallel within each batch.
                                      Defaults to False (sequential execution).
                                      When True, commands run concurrently within the same job.

            gpus_per_task (int | None, optional): Number of GPUs to assign to each command in a batch.
                                                 Defaults to None (no GPU assignment).
                                                 When set, each command gets its own GPU via CUDA_VISIBLE_DEVICES.
                                                 Example: With gpus_per_task=1 and 4 commands, each command
                                                 gets assigned GPU 0, 1, 2, or 3 respectively.
                                                 **Note**: Only works when commands are passed individually,
                                                 which happens automatically when this parameter is set.

            num_gpus (int | None, optional): Total number of GPUs available for distribution.
                                            Defaults to None (inferred from config or unlimited).
                                            Used for validation and ensuring commands don't exceed
                                            available GPU resources.

        Raises:
            FileNotFoundError: If the specified config_name does not exist in any config directory
            PermissionError: If archive_path is not writable
            ImportError: If required dependencies are not available
            
        Example:
            >>> # Basic usage with default settings
            >>> submitter = ASlurmSubmitter('my_cluster_config')

            >>> # Advanced usage with custom settings
            >>> import logging
            >>> logger = logging.getLogger(__name__)
            >>> submitter = ASlurmSubmitter(
            ...     config_name='gpu_partition',
            ...     batch_size=8,
            ...     randomize=True,
            ...     logger=logger,
            ...     dry_run=False,
            ...     archive_path='/scratch/user/slurm_jobs'
            ... )

            >>> # GPU assignment example - assign 1 GPU per command
            >>> submitter = ASlurmSubmitter(
            ...     config_name='gpu_4x_config',
            ...     batch_size=4,
            ...     gpus_per_task=1,
            ...     parallel=True
            ... )
            >>> for i in range(4):
            ...     submitter.add_command(f'python train.py --model=model{i}')
            >>> submitter.submit()  # Each command gets GPU 0, 1, 2, 3 respectively
        
        Note:
            The constructor performs initial validation of the config_name but does not
            load the full configuration until submit() is called. This allows for faster
            initialization when creating multiple submitter instances.
        """
        
        ## --- constructor arguments ---
        self.config_name = config_name
        self.batch_size = batch_size
        self.randomize = randomize
        self.logger = logger
        self.overwrite_fillers = overwrite_fillers
        self.parallel = parallel
        self.gpus_per_task = gpus_per_task
        self.num_gpus = num_gpus

        ## --- computed properties ---

        # This list will store all of the individual commands that are added to the submitter over the
        # course of its lifetime, until the `submit` method is called.
        self.commands: list[str] = []

        self.options = {
            'config_name':          config_name,
            'overwrite_fillers':    overwrite_fillers,
            'same':                 False,
            'gpus_per_task':        gpus_per_task,
            'num_gpus':             num_gpus,
            'max_tasks':            None,
            'archive_path':         archive_path,
            'dry_run':              dry_run,
            'version':              False,
        }
        self.cli = ASlurm()
        self.cli.options.update(self.options)
        self.ctx = click.Context(self.cli)
        self.ctx.obj = self.cli
        
    
    def add_command(self, command: str) -> None:
        """
        Add a command to the internal queue for later submission to SLURM.
        
        Commands are stored in the order they are added (unless randomization is enabled)
        and will be grouped into batches according to the batch_size parameter when
        submit() is called. Each command should be a complete shell command that can
        be executed independently.
        
        Args:
            command (str): A shell command to be executed in the SLURM environment.
                          Should be a complete command including all arguments and options.
                          Examples:
                          - 'python train.py --epochs=100 --lr=0.01'
                          - 'bash process_data.sh /path/to/input /path/to/output'
                          - 'conda activate myenv && python script.py'
        
        Returns:
            None
            
        Raises:
            TypeError: If command is not a string
            
        Example:
            >>> submitter = ASlurmSubmitter('gpu_config')
            >>> submitter.add_command('python experiment1.py --param=value1')
            >>> submitter.add_command('python experiment2.py --param=value2')
            >>> print(f"Queued {len(submitter.commands)} commands")
            Queued 2 commands
        
        Note:
            - Commands are not validated at addition time
            - Commands with shell operators (&&, ||, |, >) are supported
            - Multi-line commands should use proper shell syntax with line continuations
            - Environment variables in commands will be resolved in the SLURM execution context
        
        See Also:
            submit(): Submit all queued commands to SLURM
            count_jobs(): Get estimated number of SLURM jobs that will be created
        """
        self.commands.append(command)

    def _get_num_gpus(self) -> int:
        """
        Determine the number of GPUs available from configuration or parameters.

        This method attempts to determine the GPU count through multiple strategies:
        1. Use explicitly provided num_gpus parameter (highest priority)
        2. Load from config's NO_gpus field
        3. Parse from config's gres field (e.g., "gpu:4" or "gpu:full:4")
        4. Default to 1 if unable to determine

        Returns:
            int: Number of GPUs available for task distribution

        Example:
            >>> submitter = ASlurmSubmitter('haicore_4gpu', gpus_per_task=1)
            >>> submitter._get_num_gpus()
            4
        """
        # Priority 1: Explicit num_gpus parameter
        if self.num_gpus is not None:
            return self.num_gpus

        # Priority 2 & 3: Load from config
        try:
            config = self.cli.load_config(config_name=self.config_name)

            # Try NO_gpus field first
            if hasattr(config, 'NO_gpus') and config.NO_gpus is not None:
                return int(config.NO_gpus)

            # Try parsing from gres field
            if hasattr(config, 'default_fillers') and 'gres' in config.default_fillers:
                gres = config.default_fillers['gres']
                # Parse formats like "gpu:4", "gpu:full:4", etc.
                import re
                match = re.search(r'gpu:(?:\w+:)?(\d+)', gres)
                if match:
                    return int(match.group(1))

        except Exception:
            # If config loading fails, fall through to default
            pass

        # Default fallback
        self.logger.warning(
            f"Could not determine GPU count from config '{self.config_name}'. "
            f"Defaulting to 1 GPU. Consider setting num_gpus parameter explicitly."
        )
        return 1

    def submit(self):
        """
        Submit all queued commands to SLURM as one or more batch jobs.
        
        This method processes all commands in the internal queue, groups them into batches
        according to the batch_size parameter, and submits each batch as a separate SLURM job.
        The commands are executed sequentially within each batch but batches run in parallel
        on the cluster.
        
        The submission process involves:
        1. Grouping commands into batches (with optional randomization)
        2. Loading the specified AutoSlurm configuration
        3. Generating SLURM script files for each batch
        4. Submitting scripts to SLURM using sbatch (unless dry_run=True)
        5. Creating archive directories for script storage
        
        Batch Creation Logic:
        - If batch_size=1: Each command becomes its own SLURM job
        - If batch_size>1: Commands are grouped, with the last batch potentially smaller
        - If randomize=True: Commands are shuffled before batching
        
        Resource Allocation:
        - Each batch job inherits resource settings from the configuration
        - GPU allocation (if specified) is handled per the config's GRES settings
        - Memory and CPU limits apply to the entire batch, not individual commands
        
        Returns:
            None
            
        Raises:
            FileNotFoundError: If the specified config_name cannot be found
            subprocess.CalledProcessError: If sbatch command fails (when not in dry_run mode)
            PermissionError: If unable to write to archive_path
            RuntimeError: If no commands have been queued
            
        Example:
            >>> submitter = ASlurmSubmitter('gpu_config', batch_size=2)
            >>> submitter.add_command('python train1.py')
            >>> submitter.add_command('python train2.py') 
            >>> submitter.add_command('python train3.py')
            >>> submitter.submit()
            # Creates 2 SLURM jobs: 
            # Job 1: train1.py + train2.py
            # Job 2: train3.py
        
        Side Effects:
            - Creates timestamped directories in archive_path for script storage
            - Generates main_N.sh and resume_N.sh files for each batch
            - Submits jobs to SLURM queue (unless dry_run=True)
            - Clears the internal command queue after successful submission
            
        Performance Considerations:
            - Large batch sizes reduce SLURM queue overhead but increase job completion time
            - Small batch sizes maximize parallelism but may overwhelm the scheduler
            - Consider cluster policies and resource availability when choosing batch size
            
        See Also:
            add_command(): Add commands to the queue
            count_jobs(): Get estimated number of jobs before submission
            submit_batch(): Internal method for submitting individual batches
        """
        
        num_jobs = self.count_jobs()
        
        with tqdm(total=num_jobs, desc='Submitting jobs', unit='job') as pbar:
        
            batched_commands = Batched(
                self.commands, 
                batch_size=self.batch_size, 
                randomize=self.randomize
            )
            for commands in batched_commands:
                
                with redirect_stdout(io.StringIO()):
                    # This method will actually submit the given batch of commands as a single SLURM job to the 
                    # SLURM scheduler of the operating system using the existing CLI interface.
                    self.submit_batch(commands)
                
                pbar.update(1)
    
    def submit_batch(self, commands: list[str]) -> None:
        """
        Submit a single batch of commands as one SLURM job.
        
        This is an internal method called by submit() to handle individual batches.
        It combines multiple commands into a single script and submits it to SLURM
        using the configured AutoSlurm settings. Commands within a batch are executed
        sequentially on the same compute node.
        
        The method performs the following operations:
        1. Joins all commands with newline separators into a single script
        2. Invokes the underlying ASlurm CLI interface to generate SLURM scripts
        3. Submits the generated script to SLURM using sbatch
        4. Handles job ID extraction and error reporting
        
        Args:
            commands (list[str]): A list of shell commands to execute sequentially
                                 within a single SLURM job. Each command should be
                                 a complete, executable shell command.
                                 Example: ['python train.py --lr=0.01', 'python eval.py']
            
        Returns:
            None
            
        Raises:
            subprocess.CalledProcessError: If the sbatch command fails to submit the job
            ValueError: If the sbatch output format is unexpected
            RuntimeError: If the CLI interface fails to generate scripts
            
        Implementation Details:
            - Commands are separated by newlines in the generated script
            - Each command's exit status is preserved in the script
            - Failed commands will cause the entire batch job to fail
            - The batch inherits all resource settings from the AutoSlurm configuration
            
        Script Generation:
            The method uses AutoSlurm's templating system to create:
            - main_N.sh: Primary execution script with SLURM directives
            - resume_N.sh: Recovery script for restarting failed jobs
            
        Example:
            >>> # Internal usage (called by submit())
            >>> submitter = ASlurmSubmitter('config')
            >>> batch = ['python task1.py', 'python task2.py']
            >>> submitter.submit_batch(batch)
            # Submits one SLURM job executing both tasks sequentially
        
        Note:
            This is an internal method and should not typically be called directly.
            Use submit() instead, which handles batching logic and calls this method
            for each batch automatically.
            
        See Also:
            submit(): Main method for submitting all queued commands
            ASlurm.cmd_command: Underlying CLI command used for script generation
        """
        # --- 1. Determine submission method based on GPU assignment ---
        # If gpus_per_task is set, we need to pass commands individually so each can get its own
        # GPU assignment via CUDA_VISIBLE_DEVICES. Otherwise, we join commands into a single string
        # for backward compatibility.

        if self.gpus_per_task is not None:
            # --- GPU assignment mode: distribute commands across GPUs ---
            # When GPU assignment is enabled, we need to intelligently distribute commands
            # across available GPUs, ensuring each GPU processes its assigned commands
            # sequentially while all GPUs work in parallel.

            # Get the number of available GPUs
            num_gpus = self._get_num_gpus()

            # Group commands by GPU assignment (round-robin distribution)
            # Example: 8 commands, 4 GPUs
            #   GPU 0: [cmd0, cmd4]
            #   GPU 1: [cmd1, cmd5]
            #   GPU 2: [cmd2, cmd6]
            #   GPU 3: [cmd3, cmd7]
            gpu_command_groups = [[] for _ in range(num_gpus)]
            for i, command in enumerate(commands):
                gpu_idx = i % num_gpus
                gpu_command_groups[gpu_idx].append(command)

            # Create compound commands for each GPU
            # Commands within each GPU group are joined with ';' for sequential execution
            # The template will then assign different CUDA_VISIBLE_DEVICES to each group
            compound_commands = []
            for gpu_cmds in gpu_command_groups:
                if gpu_cmds:  # Only include non-empty GPU groups
                    # Join commands with ';' for sequential execution on this GPU
                    compound_commands.append(' ; '.join(gpu_cmds))

            # Build args list for CLI: ['compound_cmd1', 'cmd', 'compound_cmd2', ...]
            args = []
            for i, compound_cmd in enumerate(compound_commands):
                if i > 0:
                    args.append('cmd')
                args.append(compound_cmd)

            # Temporarily set 'same' to True to ensure all commands stay in one job
            # This prevents the CLI from splitting commands across multiple jobs
            original_same = self.options['same']
            self.options['same'] = True
            self.cli.options['same'] = True

            try:
                # Submit with compound commands
                self.ctx.invoke(self.cli.cmd_command, args=args)
            finally:
                # Restore original 'same' value
                self.options['same'] = original_same
                self.cli.options['same'] = original_same
        else:
            # --- Default mode: join commands into single string ---
            # For backward compatibility, when GPU assignment is not needed, join all commands
            # into a single string that gets executed as one task.
            if self.parallel:
                command_string: str = ' &\n '.join(commands)
            else:
                command_string: str = ' ; '.join(commands)

            # Submit using the CLI with joined command string
            self.ctx.invoke(self.cli.cmd_command, args=[command_string])

    def count_jobs(self) -> int:
        """
        Calculate the estimated number of SLURM jobs that will be created upon submission.
        
        This method provides a preview of how many individual SLURM jobs will be generated
        when submit() is called, based on the current number of queued commands and the
        configured batch_size. This is useful for planning resource usage and estimating
        cluster queue impact before submission.
        
        The calculation uses ceiling division to account for partial batches:
        - If commands % batch_size == 0: result = commands / batch_size
        - If commands % batch_size > 0: result = (commands / batch_size) + 1
        
        Returns:
            int: The number of SLURM jobs that will be created when submit() is called.
                 Returns 0 if no commands have been queued.
                 
        Example:
            >>> submitter = ASlurmSubmitter('config', batch_size=3)
            >>> submitter.count_jobs()
            0
            >>> submitter.add_command('python script1.py')
            >>> submitter.add_command('python script2.py')
            >>> submitter.count_jobs()
            1
            >>> submitter.add_command('python script3.py')
            >>> submitter.add_command('python script4.py')
            >>> submitter.count_jobs()
            2  # Jobs: [script1, script2, script3] and [script4]
        
        Use Cases:
            - **Resource Planning**: Estimate cluster resource requirements
            - **Queue Management**: Avoid overwhelming the SLURM scheduler
            - **Progress Monitoring**: Track submission progress in batch workflows
            - **Cost Estimation**: Calculate job-based billing on commercial clusters
            
        Note:
            - This method does not modify the command queue
            - The estimate is exact unless commands are modified between calling this method and submit()
            - Randomization (if enabled) does not affect the job count, only command order
            
        See Also:
            submit(): Submit all queued commands and create the estimated number of jobs
            add_command(): Add commands to the queue (affects the count)
        """
        return math.ceil(len(self.commands) / self.batch_size)



@click.group(cls=ASlurm, invoke_without_command=True)
@click.option('--config-name', '-cn', help='Config name to use for scheduling.')
@click.option('--overwrite-fillers', '-o', type=KeyValueList(), default={}, help=(
    'overwrite fillers from the config. This should be a comma-separated list of key=value pairs - '
    'e.g. time=01:00:00,mem=16G'
))
@click.option('--same', '-s', is_flag=True, help='Put all the commands into the same job.')
@click.option('--gpus-per-task', '-gpt', type=int, default=None, show_default=True, help=(
    'Number of GPUs per task. If set to a value greater than 0, the individual tasks within a job will '
    'be allocated this number of GPUs using the CUDA_VISIBLE_DEVICES environment variable. '
))
@click.option('--num-gpus', '-ng', type=int, default=None, show_default=True, help=(
    'Number of GPUs to use in total. If not set, will use all available GPUs.'
))
@click.option('--max-tasks', '-mt', type=int, default=4, show_default=True, help=(
    'Maximum number of tasks per job. If this is set and the number of commands exceeds this value, '
    'a new job will be created will be created for every `max_tasks` commands instead of putting them all into the same job.'
))
@click.option('--archive-path', type=click.Path(file_okay=False, dir_okay=True), 
    default=os.getcwd(), show_default=True, help=(
    'Path to the folder where the SLURM bash scripts will be created and archived. '
    'If not set, the current working directory from which the command is run will be used.'
))
@click.option('--dry-run', '-d', is_flag=True, help='Do not actually submit the jobs, just print the commands that would be run.')
@click.option('--version', '-v', is_flag=True, help='Show the version.')
@click.pass_context
def aslurm(ctx: click.Context,
           config_name: str,
           overwrite_fillers: dict,
           same: bool,
           gpus_per_task: int | None,
           num_gpus: int | None,
           max_tasks: int | None,
           archive_path: str,
           dry_run: bool,
           version: bool,
           ) -> None:        

    # For the --version flag we literally only print the version string and exit, much like the 
    # help flag works.
    if version:
        version_string: str = get_version()
        click.echo(version_string)
        sys.exit(0)

    ctx.obj = ctx.command
    options = {
        'config_name':          config_name,
        'overwrite_fillers':    overwrite_fillers,
        'same':                 same,
        'gpus_per_task':        gpus_per_task, 
        'num_gpus':             num_gpus,
        'max_tasks':            max_tasks,
        'archive_path':         archive_path,
        'dry_run':              dry_run,
        'version':              version
    }
    ctx.command.options.update(options)
        

if __name__ == '__main__':
    aslurm()