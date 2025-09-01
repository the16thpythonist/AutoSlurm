import os
import pytest
import tempfile
from click.testing import CliRunner
from auto_slurm.aslurmx import aslurm, ASlurmSubmitter
from auto_slurm.helpers import get_version


class TestAslurmX:
    
    def test_aslurmx_help_command(self):
        runner = CliRunner()
        result = runner.invoke(aslurm, ['--help'])
        assert result.exit_code == 0
        assert 'Options' in result.output
        assert 'Commands' in result.output
        assert 'AutoSlurm Command Line Interface' in result.output
        
    def test_aslurmx_version_command(self):
        
        version_true = get_version()
        
        runner = CliRunner()
        result = runner.invoke(aslurm, ['--version'])
        assert result.exit_code == 0
        assert version_true in result.output
        
    @pytest.mark.parametrize('config', ['haicore_1gpu', 'haicore_4gpu'])
    def test_aslurmx_generally_works(self, config: str):
        
        with tempfile.TemporaryDirectory() as temp_path:
            
            runner = CliRunner()
            result = runner.invoke(aslurm, [
                f'--archive-path={temp_path}',
                f'--config-name={config}',
                '-d',
                'cmd', 'python -c "print(True)"'
            ])
            assert result.exit_code == 0, f"Command failed: {result.output}"
        
    @pytest.mark.parametrize('max_tasks', [2, ])
    def test_aslurmx_max_commands_works(self, max_tasks: int):
        
        multiplier = 2
        num_tasks = max_tasks * multiplier
        tasks = ["cmd", "python -c 'print(True)'"] * num_tasks
        
        with tempfile.TemporaryDirectory() as temp_path:
            
            runner = CliRunner()
            result = runner.invoke(aslurm, [
                f'--archive-path={temp_path}',
                '--config-name=haicore_1gpu',
                '-d',
                '-mt', str(max_tasks),
                *tasks
            ])
            assert result.exit_code == 0, f"Command failed: {result.output}"
            
            aslurm_path = os.path.join(temp_path, '.aslurm')
            assert os.path.exists(aslurm_path), "aslurm directory was not created."
            sh_files = [
                os.path.join(root, file)
                for root, _, files in os.walk(aslurm_path)
                for file in files if file.endswith('.sh')
            ]
            # Each task creates a main and a resume script
            assert len(sh_files) == multiplier * 2


class TestASlurmSubmitter:
    """Test cases for ASlurmSubmitter class using dry_run option."""
    
    def test_aslurm_submitter_basic_dry_run(self):
        """Test basic ASlurmSubmitter functionality with dry_run."""
        with tempfile.TemporaryDirectory() as temp_path:
            submitter = ASlurmSubmitter(
                config_name='haicore_1gpu',
                dry_run=True,
                archive_path=temp_path
            )
            
            # Add some commands
            submitter.add_command('python -c "print(\'Hello World\')"')
            submitter.add_command('echo "Test command"')
            
            # Submit the commands
            submitter.submit()
            
            # Check that the archive directory was created
            aslurm_path = os.path.join(temp_path, '.aslurm')
            assert os.path.exists(aslurm_path), "Archive directory (.aslurm) was not created"
            
            # Check that script files were generated
            sh_files = []
            for root, _, files in os.walk(aslurm_path):
                for file in files:
                    if file.endswith('.sh'):
                        sh_files.append(os.path.join(root, file))
            
            # Should have at least main and resume scripts
            assert len(sh_files) >= 2, f"Expected at least 2 .sh files, found {len(sh_files)}"
            
            # Check that scripts contain our commands
            main_scripts = [f for f in sh_files if 'main' in os.path.basename(f)]
            assert len(main_scripts) >= 1, "No main script files found"
            
            # Read all main scripts and check that all commands are present
            all_script_content = ""
            for script_path in main_scripts:
                with open(script_path, 'r') as f:
                    all_script_content += f.read()
            
            assert 'Hello World' in all_script_content, "Command not found in main scripts"
            assert 'Test command' in all_script_content, "Second command not found in main scripts"

    def test_aslurm_submitter_batch_size_dry_run(self):
        """Test ASlurmSubmitter with different batch sizes in dry_run."""
        with tempfile.TemporaryDirectory() as temp_path:
            batch_size = 2
            submitter = ASlurmSubmitter(
                config_name='haicore_1gpu',
                batch_size=batch_size,
                dry_run=True,
                archive_path=temp_path
            )
            
            # Add more commands than batch size
            commands = [
                'python -c "print(1)"',
                'python -c "print(2)"',
                'python -c "print(3)"',
                'python -c "print(4)"',
                'python -c "print(5)"'
            ]
            
            for cmd in commands:
                submitter.add_command(cmd)
            
            # Verify expected job count before submission
            expected_jobs = len(commands) // batch_size + (1 if len(commands) % batch_size else 0)
            assert submitter.count_jobs() == expected_jobs
            
            submitter.submit()
            
            # Check that scripts were created
            aslurm_path = os.path.join(temp_path, '.aslurm')
            assert os.path.exists(aslurm_path)
            
            # Count script files (should have main + resume for each job)
            sh_files = []
            for _, _, files in os.walk(aslurm_path):
                sh_files.extend([f for f in files if f.endswith('.sh')])
            
            expected_script_count = expected_jobs * 2  # main + resume for each job
            assert len(sh_files) == expected_script_count, f"Expected {expected_script_count} scripts, found {len(sh_files)}"

    def test_aslurm_submitter_randomize_dry_run(self):
        """Test ASlurmSubmitter with randomize option in dry_run."""
        with tempfile.TemporaryDirectory() as temp_path:
            submitter = ASlurmSubmitter(
                config_name='haicore_1gpu',
                batch_size=3,
                randomize=True,
                dry_run=True,
                archive_path=temp_path
            )
            
            # Add commands
            commands = [f'echo "Command {i}"' for i in range(10)]
            for cmd in commands:
                submitter.add_command(cmd)
            
            submitter.submit()
            
            # Verify archive was created
            aslurm_path = os.path.join(temp_path, '.aslurm')
            assert os.path.exists(aslurm_path)
            
            # Verify scripts were created
            sh_files = []
            for _, _, files in os.walk(aslurm_path):
                sh_files.extend([f for f in files if f.endswith('.sh')])
            
            assert len(sh_files) > 0, "No script files were created"

    def test_aslurm_submitter_custom_archive_path_dry_run(self):
        """Test ASlurmSubmitter with custom archive path in dry_run."""
        with tempfile.TemporaryDirectory() as temp_base:
            custom_archive = os.path.join(temp_base, 'custom_scripts')
            os.makedirs(custom_archive, exist_ok=True)
            
            submitter = ASlurmSubmitter(
                config_name='haicore_1gpu',
                dry_run=True,
                archive_path=custom_archive
            )
            
            submitter.add_command('echo "Custom archive test"')
            submitter.submit()
            
            # Check that scripts were created in custom location
            aslurm_path = os.path.join(custom_archive, '.aslurm')
            assert os.path.exists(aslurm_path), f"Custom archive directory not created at {aslurm_path}"
            
            # Verify files exist
            sh_files = []
            for _, _, files in os.walk(aslurm_path):
                sh_files.extend([f for f in files if f.endswith('.sh')])
            
            assert len(sh_files) >= 2, "Scripts not created in custom archive location"

    def test_aslurm_submitter_no_submission_without_dry_run_verification(self):
        """Verify that dry_run prevents actual SLURM submission."""
        with tempfile.TemporaryDirectory() as temp_path:
            submitter = ASlurmSubmitter(
                config_name='haicore_1gpu',
                dry_run=True,
                archive_path=temp_path
            )
            
            submitter.add_command('echo "This should not be submitted to SLURM"')
            
            # This should complete without actually submitting to SLURM
            # (no subprocess.run with sbatch should be called)
            submitter.submit()
            
            # Scripts should still be created for inspection
            aslurm_path = os.path.join(temp_path, '.aslurm')
            assert os.path.exists(aslurm_path)

    def test_aslurm_submitter_script_content_verification(self):
        """Test that generated scripts contain expected SLURM directives and commands."""
        with tempfile.TemporaryDirectory() as temp_path:
            submitter = ASlurmSubmitter(
                config_name='haicore_1gpu',
                dry_run=True,
                archive_path=temp_path
            )
            
            test_commands = [
                'python train.py --epochs 10',
                'python evaluate.py --model checkpoint.pt'
            ]
            
            for cmd in test_commands:
                submitter.add_command(cmd)
            
            submitter.submit()
            
            # Find and read the main script
            aslurm_path = os.path.join(temp_path, '.aslurm')
            main_scripts = []
            for root, _, files in os.walk(aslurm_path):
                main_scripts.extend([
                    os.path.join(root, f) for f in files 
                    if f.startswith('main_') and f.endswith('.sh')
                ])
            
            assert len(main_scripts) >= 1, "No main script found"
            
            # Read all main scripts to check for all commands
            all_script_content = ""
            for script_path in main_scripts:
                with open(script_path, 'r') as f:
                    script_content = f.read()
                    all_script_content += script_content
                    
                    # Check for SLURM directives (these should be in the template)
                    assert '#!/bin/bash' in script_content, "Missing shebang"
                
            # Check that all our commands are present across all scripts
            for cmd in test_commands:
                assert cmd in all_script_content, f"Command '{cmd}' not found in scripts"

    @pytest.mark.parametrize('config_name', ['haicore_1gpu', 'haicore_4gpu'])
    def test_aslurm_submitter_different_configs_dry_run(self, config_name):
        """Test ASlurmSubmitter with different configurations in dry_run."""
        with tempfile.TemporaryDirectory() as temp_path:
            submitter = ASlurmSubmitter(
                config_name=config_name,
                dry_run=True,
                archive_path=temp_path
            )
            
            submitter.add_command(f'echo "Testing with {config_name}"')
            submitter.submit()
            
            # Verify archive creation
            aslurm_path = os.path.join(temp_path, '.aslurm')
            assert os.path.exists(aslurm_path)
            
            # Verify script creation
            sh_files = []
            for _, _, files in os.walk(aslurm_path):
                sh_files.extend([f for f in files if f.endswith('.sh')])
            
            assert len(sh_files) >= 2, f"Scripts not created for config {config_name}"

    def test_aslurm_submitter_empty_commands_dry_run(self):
        """Test ASlurmSubmitter behavior with no commands in dry_run."""
        with tempfile.TemporaryDirectory() as temp_path:
            submitter = ASlurmSubmitter(
                config_name='haicore_1gpu',
                dry_run=True,
                archive_path=temp_path
            )
            
            # Don't add any commands
            assert submitter.count_jobs() == 0
            
            # Submit should handle empty command list gracefully
            submitter.submit()

    def test_aslurm_submitter_count_jobs_accuracy(self):
        """Test that count_jobs returns accurate estimates."""
        submitter = ASlurmSubmitter(
            config_name='haicore_1gpu',
            batch_size=3,
            dry_run=True
        )
        
        # Test with various command counts
        test_cases = [
            (0, 0),   # No commands
            (1, 1),   # Single command
            (3, 1),   # Exact batch size
            (4, 2),   # One over batch size
            (9, 3),   # Multiple full batches
            (10, 4),  # Multiple batches with remainder
        ]
        
        for command_count, expected_jobs in test_cases:
            # Reset commands
            submitter.commands = []
            
            # Add specified number of commands
            for i in range(command_count):
                submitter.add_command(f'echo "Command {i}"')
            
            assert submitter.count_jobs() == expected_jobs, \
                f"For {command_count} commands with batch_size 3, expected {expected_jobs} jobs, got {submitter.count_jobs()}"

    def test_aslurm_submitter_archive_path_creation(self):
        """Test that archive paths are created correctly."""
        with tempfile.TemporaryDirectory() as temp_base:
            # Test with nested path that doesn't exist
            nested_path = os.path.join(temp_base, 'level1', 'level2', 'scripts')
            
            submitter = ASlurmSubmitter(
                config_name='haicore_1gpu',
                dry_run=True,
                archive_path=nested_path
            )
            
            submitter.add_command('echo "Deep path test"')
            submitter.submit()
            
            # Verify the nested structure was created
            aslurm_path = os.path.join(nested_path, '.aslurm')
            assert os.path.exists(aslurm_path), "Nested archive path was not created"
            
            # Verify scripts exist
            sh_files = []
            for _, _, files in os.walk(aslurm_path):
                sh_files.extend([f for f in files if f.endswith('.sh')])
            
            assert len(sh_files) >= 2, "Scripts not created in nested archive path"

    def test_aslurm_submitter_overwrite_fillers_dry_run(self):
        """Test that overwrite_fillers properly overrides config defaults."""
        with tempfile.TemporaryDirectory() as temp_path:
            # Custom fillers to override defaults
            custom_fillers = {
                'time': '11:00:00',
                'mem': '32G', 
                'cpus': '8'
            }
            
            submitter = ASlurmSubmitter(
                config_name='haicore_1gpu',
                dry_run=True,
                archive_path=temp_path,
                overwrite_fillers=custom_fillers
            )
            
            submitter.add_command('echo "Testing overwrite fillers"')
            submitter.submit()
            
            # Find and read the main script to verify overwrite_fillers were applied
            aslurm_path = os.path.join(temp_path, '.aslurm')
            assert os.path.exists(aslurm_path), "Archive directory was not created"
            
            main_scripts = []
            for root, _, files in os.walk(aslurm_path):
                main_scripts.extend([
                    os.path.join(root, f) for f in files 
                    if f.startswith('main_') and f.endswith('.sh')
                ])
            
            assert len(main_scripts) >= 1, "No main script found"
            
            # Read the script content and check for overwritten values
            with open(main_scripts[0], 'r') as f:
                script_content = f.read()
            
            # Check that our custom time value appears in the SLURM directives
            assert '11:00:00' in script_content, "Custom time value '11:00:00' not found in script"
            # The exact format depends on the template, but these values should appear somewhere
            # We check for the values as they might be used in different SLURM directive formats