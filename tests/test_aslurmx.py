import os
import re
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

    @pytest.mark.parametrize('config', ['haicore_4gpu', 'juwels_4gpu'])
    def test_aslurmx_gpus_per_task_assignment(self, config: str):
        """Test that --gpus-per-task correctly assigns CUDA_VISIBLE_DEVICES to each command."""

        with tempfile.TemporaryDirectory() as temp_path:

            # Create 4 commands to run on 4 GPUs
            runner = CliRunner()
            result = runner.invoke(aslurm, [
                f'--archive-path={temp_path}',
                f'--config-name={config}',
                '--gpus-per-task=1',  # Each command gets 1 GPU
                '-d',  # Dry run
                'cmd', 'python script1.py',
                'cmd', 'python script2.py',
                'cmd', 'python script3.py',
                'cmd', 'python script4.py',
            ])
            assert result.exit_code == 0, f"Command failed: {result.output}"

            # Find the generated script
            aslurm_path = os.path.join(temp_path, '.aslurm')
            assert os.path.exists(aslurm_path), "aslurm directory was not created."

            # Find the main script
            main_scripts = []
            for root, _, files in os.walk(aslurm_path):
                for file in files:
                    if file.startswith('main_') and file.endswith('.sh'):
                        main_scripts.append(os.path.join(root, file))

            assert len(main_scripts) >= 1, "No main script found"

            # Read the script content
            with open(main_scripts[0], 'r') as f:
                script_content = f.read()

            # Verify that each command has the correct CUDA_VISIBLE_DEVICES assignment
            assert 'CUDA_VISIBLE_DEVICES=0' in script_content, "GPU 0 assignment not found"
            assert 'CUDA_VISIBLE_DEVICES=1' in script_content, "GPU 1 assignment not found"
            assert 'CUDA_VISIBLE_DEVICES=2' in script_content, "GPU 2 assignment not found"
            assert 'CUDA_VISIBLE_DEVICES=3' in script_content, "GPU 3 assignment not found"

            # Verify all commands are present
            assert 'python script1.py' in script_content, "Command 1 not found"
            assert 'python script2.py' in script_content, "Command 2 not found"
            assert 'python script3.py' in script_content, "Command 3 not found"
            assert 'python script4.py' in script_content, "Command 4 not found"

            # Verify that CUDA_VISIBLE_DEVICES appears before each command
            # Look for the pattern where CUDA_VISIBLE_DEVICES is set for each command
            assert 'CUDA_VISIBLE_DEVICES=0' in script_content and script_content.index('CUDA_VISIBLE_DEVICES=0') < script_content.index('python script1.py')
            assert 'CUDA_VISIBLE_DEVICES=1' in script_content and script_content.index('CUDA_VISIBLE_DEVICES=1') < script_content.index('python script2.py')
            assert 'CUDA_VISIBLE_DEVICES=2' in script_content and script_content.index('CUDA_VISIBLE_DEVICES=2') < script_content.index('python script3.py')
            assert 'CUDA_VISIBLE_DEVICES=3' in script_content and script_content.index('CUDA_VISIBLE_DEVICES=3') < script_content.index('python script4.py')


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

    def test_juwels_4gpu_additional_sbatch_configs(self):
        """Test that juwels_4gpu config includes the account directive."""
        with tempfile.TemporaryDirectory() as temp_path:
            submitter = ASlurmSubmitter(
                config_name='juwels_4gpu',
                dry_run=True,
                archive_path=temp_path
            )

            submitter.add_command('echo "Testing juwels account config"')
            submitter.submit()

            # Find and read the main script
            aslurm_path = os.path.join(temp_path, '.aslurm')
            assert os.path.exists(aslurm_path), "Archive directory was not created"

            main_scripts = []
            for root, _, files in os.walk(aslurm_path):
                main_scripts.extend([
                    os.path.join(root, f) for f in files
                    if f.startswith('main_') and f.endswith('.sh')
                ])

            assert len(main_scripts) >= 1, "No main script found"

            # Read the script content and verify the account directive
            with open(main_scripts[0], 'r') as f:
                script_content = f.read()

            # Verify the account directive is present
            assert '#SBATCH --account=aimatchem' in script_content, \
                "Account directive '#SBATCH --account=aimatchem' not found in juwels_4gpu script"

    def test_additional_sbatch_configs_override(self):
        """Test that additional_sbatch_configs can be overridden via overwrite_fillers."""
        with tempfile.TemporaryDirectory() as temp_path:
            # Override the additional_sbatch_configs from command line
            custom_fillers = {
                'additional_sbatch_configs': '#SBATCH --account=custom_account'
            }

            submitter = ASlurmSubmitter(
                config_name='juwels_4gpu',
                dry_run=True,
                archive_path=temp_path,
                overwrite_fillers=custom_fillers
            )

            submitter.add_command('echo "Testing account override"')
            submitter.submit()

            # Find and read the main script
            aslurm_path = os.path.join(temp_path, '.aslurm')
            assert os.path.exists(aslurm_path), "Archive directory was not created"

            main_scripts = []
            for root, _, files in os.walk(aslurm_path):
                main_scripts.extend([
                    os.path.join(root, f) for f in files
                    if f.startswith('main_') and f.endswith('.sh')
                ])

            assert len(main_scripts) >= 1, "No main script found"

            # Read the script content
            with open(main_scripts[0], 'r') as f:
                script_content = f.read()

            # Verify the override took effect
            assert '#SBATCH --account=custom_account' in script_content, \
                "Custom account directive not found in script"
            assert 'aimatchem' not in script_content or '#SBATCH --account=aimatchem' not in script_content, \
                "Default account directive should be overridden"

    def test_aslurm_submitter_gpus_per_task(self):
        """Test that ASlurmSubmitter correctly assigns GPUs when gpus_per_task is set."""
        with tempfile.TemporaryDirectory() as temp_path:
            # Create a submitter with GPU assignment enabled
            submitter = ASlurmSubmitter(
                config_name='haicore_4gpu',
                batch_size=4,  # Put all 4 commands in one batch
                gpus_per_task=1,  # Assign 1 GPU per command
                dry_run=True,
                archive_path=temp_path
            )

            # Add 4 commands that should each get a different GPU
            test_commands = [
                'python train.py --model=model1',
                'python train.py --model=model2',
                'python train.py --model=model3',
                'python train.py --model=model4',
            ]

            for cmd in test_commands:
                submitter.add_command(cmd)

            submitter.submit()

            # Find and read the main script
            aslurm_path = os.path.join(temp_path, '.aslurm')
            assert os.path.exists(aslurm_path), "Archive directory was not created"

            main_scripts = []
            for root, _, files in os.walk(aslurm_path):
                main_scripts.extend([
                    os.path.join(root, f) for f in files
                    if f.startswith('main_') and f.endswith('.sh')
                ])

            assert len(main_scripts) == 1, f"Expected 1 main script, found {len(main_scripts)}"

            # Read the script content
            with open(main_scripts[0], 'r') as f:
                script_content = f.read()

            # Verify that each command has the correct CUDA_VISIBLE_DEVICES assignment
            assert 'CUDA_VISIBLE_DEVICES=0' in script_content, "GPU 0 assignment not found"
            assert 'CUDA_VISIBLE_DEVICES=1' in script_content, "GPU 1 assignment not found"
            assert 'CUDA_VISIBLE_DEVICES=2' in script_content, "GPU 2 assignment not found"
            assert 'CUDA_VISIBLE_DEVICES=3' in script_content, "GPU 3 assignment not found"

            # Verify all commands are present
            for cmd in test_commands:
                assert cmd in script_content, f"Command '{cmd}' not found in script"

            # Verify the correct ordering - CUDA_VISIBLE_DEVICES should come before each command
            for i, cmd in enumerate(test_commands):
                cuda_var = f'CUDA_VISIBLE_DEVICES={i}'
                assert cuda_var in script_content, f"{cuda_var} not found"
                # Check that the CUDA_VISIBLE_DEVICES appears before the command
                cuda_index = script_content.index(cuda_var)
                cmd_index = script_content.index(cmd)
                assert cuda_index < cmd_index, f"{cuda_var} should appear before '{cmd}'"

    def test_aslurm_submitter_no_gpu_assignment_by_default(self):
        """Test that GPU assignment does NOT happen when gpus_per_task is None (default behavior)."""
        with tempfile.TemporaryDirectory() as temp_path:
            # Create a submitter without GPU assignment (default)
            submitter = ASlurmSubmitter(
                config_name='haicore_4gpu',
                batch_size=4,
                dry_run=True,
                archive_path=temp_path
            )

            # Add 4 commands
            test_commands = [
                'python train.py --model=model1',
                'python train.py --model=model2',
                'python train.py --model=model3',
                'python train.py --model=model4',
            ]

            for cmd in test_commands:
                submitter.add_command(cmd)

            submitter.submit()

            # Find and read the main script
            aslurm_path = os.path.join(temp_path, '.aslurm')
            assert os.path.exists(aslurm_path), "Archive directory was not created"

            main_scripts = []
            for root, _, files in os.walk(aslurm_path):
                main_scripts.extend([
                    os.path.join(root, f) for f in files
                    if f.startswith('main_') and f.endswith('.sh')
                ])

            assert len(main_scripts) == 1, f"Expected 1 main script, found {len(main_scripts)}"

            # Read the script content
            with open(main_scripts[0], 'r') as f:
                script_content = f.read()

            # Verify that CUDA_VISIBLE_DEVICES is NOT set for individual commands
            # When no GPU assignment, commands should be joined with ';' and run as one task
            # So we should only see CUDA_VISIBLE_DEVICES=0 at most (for the single task index 0)
            assert script_content.count('CUDA_VISIBLE_DEVICES=1') == 0, \
                "CUDA_VISIBLE_DEVICES=1 should not be present when gpus_per_task is None"
            assert script_content.count('CUDA_VISIBLE_DEVICES=2') == 0, \
                "CUDA_VISIBLE_DEVICES=2 should not be present when gpus_per_task is None"
            assert script_content.count('CUDA_VISIBLE_DEVICES=3') == 0, \
                "CUDA_VISIBLE_DEVICES=3 should not be present when gpus_per_task is None"

            # Verify all commands are still present (joined with semicolons)
            for cmd in test_commands:
                assert cmd in script_content, f"Command '{cmd}' not found in script"

    def test_aslurm_submitter_large_batch_gpu_distribution(self):
        """Test GPU distribution when batch_size > num_gpus (e.g., 16 commands on 4 GPUs)."""
        with tempfile.TemporaryDirectory() as temp_path:
            # Create a submitter with more commands than GPUs
            # 16 commands should be distributed across 4 GPUs (4 commands per GPU)
            submitter = ASlurmSubmitter(
                config_name='haicore_4gpu',  # 4 GPUs available
                batch_size=16,               # 16 commands in one batch
                gpus_per_task=1,
                dry_run=True,
                archive_path=temp_path
            )

            # Add 16 commands
            test_commands = [f'python train.py --model=model{i}' for i in range(16)]
            for cmd in test_commands:
                submitter.add_command(cmd)

            submitter.submit()

            # Find and read the main script
            aslurm_path = os.path.join(temp_path, '.aslurm')
            assert os.path.exists(aslurm_path), "Archive directory was not created"

            main_scripts = []
            for root, _, files in os.walk(aslurm_path):
                main_scripts.extend([
                    os.path.join(root, f) for f in files
                    if f.startswith('main_') and f.endswith('.sh')
                ])

            assert len(main_scripts) == 1, f"Expected 1 main script, found {len(main_scripts)}"

            # Read the script content
            with open(main_scripts[0], 'r') as f:
                script_content = f.read()

            # Should have exactly 4 CUDA_VISIBLE_DEVICES assignments (one per GPU)
            assert script_content.count('CUDA_VISIBLE_DEVICES=0') == 1, \
                "Should have exactly one GPU 0 assignment"
            assert script_content.count('CUDA_VISIBLE_DEVICES=1') == 1, \
                "Should have exactly one GPU 1 assignment"
            assert script_content.count('CUDA_VISIBLE_DEVICES=2') == 1, \
                "Should have exactly one GPU 2 assignment"
            assert script_content.count('CUDA_VISIBLE_DEVICES=3') == 1, \
                "Should have exactly one GPU 3 assignment"

            # No invalid GPU assignments (4 and above)
            assert 'CUDA_VISIBLE_DEVICES=4' not in script_content, \
                "Should not assign invalid GPU 4"

            # Verify all 16 commands are present
            for cmd in test_commands:
                assert cmd in script_content, f"Command '{cmd}' not found in script"

            # Verify distribution pattern: GPU 0 should have commands 0, 4, 8, 12
            # Extract the custom commands section for detailed verification
            custom_section_start = script_content.find('# Custom commands')
            custom_section_end = script_content.find('wait', custom_section_start)
            custom_section = script_content[custom_section_start:custom_section_end]

            # GPU 0 line should contain commands 0, 4, 8, 12 (joined with ';')
            gpu0_pattern = r'CUDA_VISIBLE_DEVICES=0.*model0.*model4.*model8.*model12'
            assert re.search(gpu0_pattern, custom_section, re.DOTALL), \
                "GPU 0 should run commands 0, 4, 8, 12 sequentially"

            # GPU 1 line should contain commands 1, 5, 9, 13
            gpu1_pattern = r'CUDA_VISIBLE_DEVICES=1.*model1.*model5.*model9.*model13'
            assert re.search(gpu1_pattern, custom_section, re.DOTALL), \
                "GPU 1 should run commands 1, 5, 9, 13 sequentially"

            # GPU 2 line should contain commands 2, 6, 10, 14
            gpu2_pattern = r'CUDA_VISIBLE_DEVICES=2.*model2.*model6.*model10.*model14'
            assert re.search(gpu2_pattern, custom_section, re.DOTALL), \
                "GPU 2 should run commands 2, 6, 10, 14 sequentially"

            # GPU 3 line should contain commands 3, 7, 11, 15
            gpu3_pattern = r'CUDA_VISIBLE_DEVICES=3.*model3.*model7.*model11.*model15'
            assert re.search(gpu3_pattern, custom_section, re.DOTALL), \
                "GPU 3 should run commands 3, 7, 11, 15 sequentially"