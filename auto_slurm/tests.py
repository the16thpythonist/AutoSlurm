import unittest
from auto_slurm.aslurm import (
    build_commands_str,
    create_slurm_job_files,
)
from auto_slurm.helpers import expand_commands


class TestSlurmScript(unittest.TestCase):

    def test_expand_commands_brackets(self):
        commands = ["python train.py --lr <[0.01,0.1]> --batch_size <[32,64]>"]
        expected = [
            "python train.py --lr 0.01 --batch_size 32",
            "python train.py --lr 0.1 --batch_size 64",
        ]
        self.assertEqual(expand_commands(commands), expected)

    def test_expand_commands_braces(self):
        commands = ["python train.py --lr <{0.01,0.1}> --batch_size <{32,64}>"]
        expected = [
            "python train.py --lr 0.01 --batch_size 32",
            "python train.py --lr 0.01 --batch_size 64",
            "python train.py --lr 0.1 --batch_size 32",
            "python train.py --lr 0.1 --batch_size 64",
        ]
        self.assertEqual(expand_commands(commands), expected)

    def test_build_commands_str(self):
        commands = ["python train.py --lr 0.01", "python train.py --lr 0.1"]
        job_start_task_index = 0
        gpus_per_task = 1

        command_str = build_commands_str(commands, job_start_task_index, gpus_per_task)

        self.assertIn(
            "SLURM_SUBMIT_TASK_INDEX=0 CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01",
            command_str,
        )
        self.assertIn(
            "SLURM_SUBMIT_TASK_INDEX=1 CUDA_VISIBLE_DEVICES=1 python train.py --lr 0.1",
            command_str,
        )

    def test_create_slurm_job_files(self):
        template = "#SBATCH --job-name=<job_name>\nsomecommand > <output_file>"
        fillers = {"job_name": "<inner_filler>", "output_file": "test.out"}
        global_fillers = {"inner_filler": "test_job"}

        create_slurm_job_files(
            "main.sh",
            "resume.sh",
            0,
            template,
            fillers,
            global_fillers,
            commands=["echo test0", "echo test1"],
            gpus_per_task=2,
        )

        # Read the files:
        with open("main.sh", "r") as f:
            main_contents = f.read()
        with open("resume.sh", "r") as f:
            resume_contents = f.read()

        self.assertIn("#SBATCH --job-name=test_job", main_contents)
        self.assertIn("#SBATCH --job-name=test_job", resume_contents)
        self.assertIn("echo test0", main_contents)
        self.assertIn("echo test1", main_contents)
        self.assertIn("eval", resume_contents)
        self.assertIn("CUDA_VISIBLE_DEVICES=0,1", main_contents)
        self.assertIn("CUDA_VISIBLE_DEVICES=0,1", resume_contents)
        self.assertIn("CUDA_VISIBLE_DEVICES=2,3", main_contents)
        self.assertIn("CUDA_VISIBLE_DEVICES=2,3", resume_contents)

    def test_expand_commands_invalid_syntax(self):
        commands = ["python train.py --lr <[0.01,0.1]> --batch_size <{32,64}>"]
        with self.assertRaises(ValueError):
            expand_commands(commands)

        commands = ["python train.py --lr <[0.01,0.1,1.0]> --batch_size <[32,64]>"]
        with self.assertRaises(ValueError):
            expand_commands(commands)

    def test_build_commands_no_gpus(self):
        commands = ["python train.py --lr 0.01"]
        command_str = build_commands_str(commands, 0, None)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", command_str)


# ============================================================================
# Tests for aslurmx.py features (hostname detection and interactive command)
# ============================================================================

import subprocess
from unittest.mock import patch, MagicMock
import pytest
from click.testing import CliRunner


class TestHostnameConfigDetection:
    """Tests for automatic hostname-based config detection."""

    @pytest.fixture
    def aslurm_instance(self):
        """Create an ASlurm instance with mocked general_config."""
        from auto_slurm.aslurmx import ASlurm
        instance = ASlurm()
        return instance

    def test_detect_config_single_match(self, aslurm_instance):
        """Test that a single matching hostname returns the correct config."""
        aslurm_instance.general_config.hostname_config_mappings = {
            r"^hkn.*": "haicore_1gpu",
            r"^uc2n.*": "bwuni_1gpu",
        }

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="hkn1234\n", returncode=0)
            result = aslurm_instance.detect_config_from_hostname()

        assert result == "haicore_1gpu"

    def test_detect_config_different_cluster(self, aslurm_instance):
        """Test detection for a different cluster hostname."""
        aslurm_instance.general_config.hostname_config_mappings = {
            r"^hkn.*": "haicore_1gpu",
            r"^uc2n.*": "bwuni_1gpu",
        }

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="uc2n0042\n", returncode=0)
            result = aslurm_instance.detect_config_from_hostname()

        assert result == "bwuni_1gpu"

    def test_detect_config_no_match_raises_error(self, aslurm_instance):
        """Test that no matching hostname raises ClickException."""
        import click
        aslurm_instance.general_config.hostname_config_mappings = {
            r"^hkn.*": "haicore_1gpu",
        }

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="unknown-host\n", returncode=0)
            with pytest.raises(click.ClickException) as exc_info:
                aslurm_instance.detect_config_from_hostname()

        assert "No config found for hostname" in str(exc_info.value)
        assert "unknown-host" in str(exc_info.value)

    def test_detect_config_multiple_matches_raises_error(self, aslurm_instance):
        """Test that multiple matching hostnames raises ClickException."""
        import click
        aslurm_instance.general_config.hostname_config_mappings = {
            r"^hkn.*": "haicore_1gpu",
            r"^hkn12.*": "haicore_4gpu",  # Overlapping pattern
        }

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="hkn1234\n", returncode=0)
            with pytest.raises(click.ClickException) as exc_info:
                aslurm_instance.detect_config_from_hostname()

        assert "Multiple configs match" in str(exc_info.value)

    def test_detect_config_empty_mappings_raises_error(self, aslurm_instance):
        """Test that empty hostname_config_mappings raises ClickException."""
        import click
        aslurm_instance.general_config.hostname_config_mappings = {}

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="hkn1234\n", returncode=0)
            with pytest.raises(click.ClickException) as exc_info:
                aslurm_instance.detect_config_from_hostname()

        assert "No hostname_config_mappings defined" in str(exc_info.value)

    def test_detect_config_hostname_command_failure(self, aslurm_instance):
        """Test that hostname command failure raises ClickException."""
        import click
        aslurm_instance.general_config.hostname_config_mappings = {
            r"^hkn.*": "haicore_1gpu",
        }

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "hostname")
            with pytest.raises(click.ClickException) as exc_info:
                aslurm_instance.detect_config_from_hostname()

        assert "Failed to get hostname" in str(exc_info.value)


class TestInteractiveCommand:
    """Tests for the interactive command."""

    @pytest.fixture
    def runner(self):
        """Create a Click test runner."""
        return CliRunner()

    def test_interactive_command_dry_run(self, runner, tmp_path):
        """Test interactive command in dry run mode creates script."""
        from auto_slurm.aslurmx import aslurm

        with patch("auto_slurm.aslurmx.ASlurm.load_config") as mock_load:
            mock_config = MagicMock()
            mock_config.default_fillers = {"partition": "test", "time": "01:00:00"}
            mock_load.return_value = mock_config

            result = runner.invoke(
                aslurm,
                ["-cn", "test_config", "--dry-run", f"--archive-path={tmp_path}", "interactive"],
            )

        assert result.exit_code == 0
        assert "Dry run" in result.output
        assert "main_interactive.sh" in result.output

    def test_interactive_command_creates_sleep_loop(self, runner, tmp_path):
        """Test that interactive command creates script with sleep loop."""
        from auto_slurm.aslurmx import aslurm

        with patch("auto_slurm.aslurmx.ASlurm.load_config") as mock_load:
            mock_config = MagicMock()
            mock_config.default_fillers = {"partition": "test", "time": "01:00:00"}
            mock_load.return_value = mock_config

            result = runner.invoke(
                aslurm,
                ["-cn", "test_config", "--dry-run", f"--archive-path={tmp_path}", "interactive"],
            )

        # Find and read the created script
        aslurm_dirs = list(tmp_path.glob(".aslurm/*"))
        assert len(aslurm_dirs) == 1

        main_script = aslurm_dirs[0] / "main_interactive.sh"
        assert main_script.exists()

        content = main_script.read_text()
        assert "while true" in content
        assert "sleep 10000" in content

    def test_interactive_with_exclude_nodes(self, runner, tmp_path):
        """Test interactive command respects exclude nodes option."""
        from auto_slurm.aslurmx import aslurm

        with patch("auto_slurm.aslurmx.ASlurm.load_config") as mock_load:
            with patch("subprocess.run") as mock_run:
                mock_config = MagicMock()
                mock_config.default_fillers = {"partition": "test"}
                mock_load.return_value = mock_config
                mock_run.return_value = MagicMock(
                    stdout="Submitted batch job 12345\n", returncode=0
                )

                result = runner.invoke(
                    aslurm,
                    [
                        "-cn", "test_config",
                        "--exclude=node01,node02",
                        f"--archive-path={tmp_path}",
                        "interactive",
                    ],
                )

        # Verify sbatch was called with exclude option
        sbatch_calls = [call for call in mock_run.call_args_list
                       if 'sbatch' in str(call)]
        assert len(sbatch_calls) > 0
        sbatch_call = sbatch_calls[0]
        assert "--exclude=node01,node02" in sbatch_call[0][0]

    def test_interactive_prints_job_instructions(self, runner, tmp_path):
        """Test that interactive command prints srun and scancel instructions."""
        from auto_slurm.aslurmx import aslurm

        with patch("auto_slurm.aslurmx.ASlurm.load_config") as mock_load:
            with patch("subprocess.run") as mock_run:
                mock_config = MagicMock()
                mock_config.default_fillers = {"partition": "test"}
                mock_load.return_value = mock_config
                mock_run.return_value = MagicMock(
                    stdout="Submitted batch job 12345\n", returncode=0
                )

                result = runner.invoke(
                    aslurm,
                    ["-cn", "test_config", f"--archive-path={tmp_path}", "interactive"],
                )

        assert "12345" in result.output
        assert "srun --jobid 12345 --pty bash" in result.output
        assert "scancel 12345" in result.output

    def test_interactive_with_auto_config_detection(self, runner, tmp_path):
        """Test interactive command works with automatic config detection."""
        from auto_slurm.aslurmx import aslurm

        with patch("auto_slurm.aslurmx.ASlurm.detect_config_from_hostname") as mock_detect:
            with patch("auto_slurm.aslurmx.ASlurm.load_config") as mock_load:
                mock_detect.return_value = "auto_detected_config"
                mock_config = MagicMock()
                mock_config.default_fillers = {"partition": "test"}
                mock_load.return_value = mock_config

                result = runner.invoke(
                    aslurm,
                    ["--dry-run", f"--archive-path={tmp_path}", "interactive"],
                )

        mock_detect.assert_called_once()
        assert "auto_detected_config" in result.output


class TestCommandExpansionIntegration:
    """Tests for command expansion (sweep syntax) integration in aslurmx."""

    @pytest.fixture
    def runner(self):
        """Create a Click test runner."""
        return CliRunner()

    def test_cmd_expands_bracket_syntax(self, runner, tmp_path):
        """Test that cmd command expands <[...]> paired list syntax."""
        from auto_slurm.aslurmx import aslurm

        with patch("auto_slurm.aslurmx.ASlurm.load_config") as mock_load:
            mock_config = MagicMock()
            mock_config.default_fillers = {"partition": "test", "time": "01:00:00"}
            mock_load.return_value = mock_config

            result = runner.invoke(
                aslurm,
                [
                    "-cn", "test_config",
                    "--dry-run",
                    f"--archive-path={tmp_path}",
                    "cmd", "python train.py --lr=<[0.1, 0.01]> --bs=<[16, 32]>",
                ],
            )

        # Should expand to 2 commands (paired/zipped)
        assert "preparing to submit 2 commands" in result.output

    def test_cmd_expands_brace_syntax(self, runner, tmp_path):
        """Test that cmd command expands <{...}> grid search syntax."""
        from auto_slurm.aslurmx import aslurm

        with patch("auto_slurm.aslurmx.ASlurm.load_config") as mock_load:
            mock_config = MagicMock()
            mock_config.default_fillers = {"partition": "test", "time": "01:00:00"}
            mock_load.return_value = mock_config

            result = runner.invoke(
                aslurm,
                [
                    "-cn", "test_config",
                    "--dry-run",
                    f"--archive-path={tmp_path}",
                    "cmd", "python train.py --lr=<{0.1, 0.01}> --bs=<{16, 32}>",
                ],
            )

        # Should expand to 4 commands (2x2 grid)
        assert "preparing to submit 4 commands" in result.output

    def test_cmd_expansion_with_cmdnx_repetition(self, runner, tmp_path):
        """Test that cmdNx repetition works together with sweep expansion."""
        from auto_slurm.aslurmx import aslurm

        with patch("auto_slurm.aslurmx.ASlurm.load_config") as mock_load:
            mock_config = MagicMock()
            mock_config.default_fillers = {"partition": "test", "time": "01:00:00"}
            mock_load.return_value = mock_config

            # cmd2x with 2-value grid = 2 repetitions * 2 values = 4 commands
            result = runner.invoke(
                aslurm,
                [
                    "-cn", "test_config",
                    "--dry-run",
                    f"--archive-path={tmp_path}",
                    "cmd2x", "python train.py --lr=<{0.1, 0.01}>",
                ],
            )

        # 2 repetitions * 2 grid values = 4 commands
        assert "preparing to submit 4 commands" in result.output

    def test_cmd_creates_expanded_scripts(self, runner, tmp_path):
        """Test that expanded commands appear in the generated SLURM script."""
        from auto_slurm.aslurmx import aslurm

        with patch("auto_slurm.aslurmx.ASlurm.load_config") as mock_load:
            mock_config = MagicMock()
            mock_config.default_fillers = {"partition": "test", "time": "01:00:00"}
            mock_load.return_value = mock_config

            result = runner.invoke(
                aslurm,
                [
                    "-cn", "test_config",
                    "--dry-run",
                    "--same",  # Put all in same job so we can check the script
                    f"--archive-path={tmp_path}",
                    "cmd", "python train.py --lr=<[0.1, 0.01]>",
                ],
            )

        # Find and read the created script
        aslurm_dirs = list(tmp_path.glob(".aslurm/*"))
        assert len(aslurm_dirs) == 1

        main_script = aslurm_dirs[0] / "main_0.sh"
        assert main_script.exists()

        content = main_script.read_text()
        # Both expanded commands should be in the script
        assert "--lr=0.1" in content
        assert "--lr=0.01" in content


class TestASlurmSubmitterExpansion:
    """Tests for command expansion in ASlurmSubmitter."""

    def test_submitter_count_jobs_with_expansion(self):
        """Test that count_jobs accounts for sweep expansion."""
        from auto_slurm.aslurmx import ASlurmSubmitter

        with patch("auto_slurm.aslurmx.ASlurm.load_config"):
            submitter = ASlurmSubmitter(
                config_name="test_config",
                batch_size=2,
                dry_run=True,
            )

            # Add command with grid syntax that expands to 4 commands
            submitter.add_command("python train.py --lr=<{0.1, 0.01}> --bs=<{16, 32}>")

            # 4 expanded commands / batch_size 2 = 2 jobs
            assert submitter.count_jobs() == 2

    def test_submitter_count_jobs_with_paired_expansion(self):
        """Test that count_jobs accounts for paired list expansion."""
        from auto_slurm.aslurmx import ASlurmSubmitter

        with patch("auto_slurm.aslurmx.ASlurm.load_config"):
            submitter = ASlurmSubmitter(
                config_name="test_config",
                batch_size=1,
                dry_run=True,
            )

            # Add command with paired syntax that expands to 3 commands
            submitter.add_command("python train.py --lr=<[0.1, 0.01, 0.001]>")

            # 3 expanded commands / batch_size 1 = 3 jobs
            assert submitter.count_jobs() == 3

    def test_submitter_no_expansion_without_syntax(self):
        """Test that commands without sweep syntax are not modified."""
        from auto_slurm.aslurmx import ASlurmSubmitter

        with patch("auto_slurm.aslurmx.ASlurm.load_config"):
            submitter = ASlurmSubmitter(
                config_name="test_config",
                batch_size=1,
                dry_run=True,
            )

            submitter.add_command("python train.py --lr=0.1")
            submitter.add_command("python train.py --lr=0.01")

            # 2 commands, no expansion
            assert submitter.count_jobs() == 2


if __name__ == "__main__":
    unittest.main()
