"""
Layer 3: Bash execution integration tests for resume/chain job functionality.

These tests actually execute the generated bash scripts (or relevant portions)
in a controlled environment to verify that the chaining logic works end-to-end.

Key setup:
- A mock `sbatch` script is placed on $PATH that logs calls and prints
  "Submitted batch job <id>" (the format SLURM uses).
- SLURM_JOB_ID is set as an environment variable.
- Fake .resume files are created to simulate what write_resume_file() does.

NOT run by default. Run explicitly with:
    pytest -m integration tests/test_resume_integration.py
"""

import os
import re
import stat
import subprocess
import tempfile

import pytest
from click.testing import CliRunner

from auto_slurm.aslurmx import aslurm


@pytest.mark.integration
class TestResumeIntegration:
    """Integration tests that execute generated bash scripts to verify resume chaining."""

    FAKE_SLURM_ID = "12345"

    @pytest.fixture
    def workspace(self, tmp_path):
        """
        Set up a workspace with:
        - A mock sbatch on PATH that logs every call to sbatch_calls.log
        - A temp directory to use as archive-path
        - Environment dict with SLURM_JOB_ID and modified PATH
        """
        mock_bin = tmp_path / "mock_bin"
        mock_bin.mkdir()

        sbatch_log = tmp_path / "sbatch_calls.log"

        # Mock sbatch: logs args and prints SLURM-like output
        mock_sbatch = mock_bin / "sbatch"
        mock_sbatch.write_text(
            f'#!/bin/bash\n'
            f'echo "$@" >> "{sbatch_log}"\n'
            f'echo "Submitted batch job 99999"\n'
        )
        mock_sbatch.chmod(mock_sbatch.stat().st_mode | stat.S_IEXEC)

        env = os.environ.copy()
        env["PATH"] = str(mock_bin) + ":" + env["PATH"]
        env["SLURM_JOB_ID"] = self.FAKE_SLURM_ID

        return {
            "tmp_path": tmp_path,
            "sbatch_log": sbatch_log,
            "env": env,
        }

    def _generate_scripts(self, tmp_path):
        """
        Helper: invoke aslurmx in dry-run mode to generate scripts.
        Returns the directory containing the generated .sh files.
        """
        runner = CliRunner()
        result = runner.invoke(aslurm, [
            f'--archive-path={str(tmp_path)}',
            '--config-name=haicore_1gpu',
            '-d',
            'cmd', 'echo hello',
        ])
        assert result.exit_code == 0, f"Script generation failed: {result.output}"

        # Find the timestamped script directory
        aslurm_path = tmp_path / ".aslurm"
        script_dirs = [
            os.path.join(str(aslurm_path), d)
            for d in os.listdir(str(aslurm_path))
            if os.path.isdir(os.path.join(str(aslurm_path), d))
        ]
        assert len(script_dirs) == 1, f"Expected 1 script dir, found {len(script_dirs)}"
        return script_dirs[0]

    def _extract_chaining_block(self, script_content):
        """
        Extract the chaining block from a generated script.

        The chaining block starts after `wait` / `sleep` and contains the
        `compgen ... .resume` check. We extract this so we can run it in
        isolation without executing the actual commands or sourcing .bashrc.
        """
        # Look for the pattern: everything from "if compgen" to the matching "fi"
        # This covers the chain-trigger logic in both main and resume scripts
        match = re.search(
            r'(if compgen.*?fi)',
            script_content,
            re.DOTALL,
        )
        assert match is not None, (
            "Could not find chaining block (if compgen...fi) in script:\n"
            + script_content
        )
        return match.group(1)

    def test_resume_triggered_when_resume_files_exist(self, workspace):
        """
        When .resume files exist for the current SLURM_JOB_ID, the main script's
        chaining block should call sbatch on the resume script.
        """
        tmp_path = workspace["tmp_path"]
        script_dir = self._generate_scripts(tmp_path)

        # Read the main script
        main_path = os.path.join(script_dir, "main_0.sh")
        assert os.path.exists(main_path), "main_0.sh not found"
        with open(main_path) as f:
            main_content = f.read()

        # Create a fake .resume file where write_resume_file() would put it
        aslurm_dir = os.path.join(str(tmp_path), ".aslurm")
        resume_file = os.path.join(aslurm_dir, f"{self.FAKE_SLURM_ID}_0.resume")
        with open(resume_file, 'w') as f:
            f.write("echo resumed_successfully")

        # Extract and run the chaining block
        chain_block = self._extract_chaining_block(main_content)

        # We need to set SCRIPT_DIR to the actual script directory since the
        # template uses it to locate the resume script
        test_script = (
            f'#!/bin/bash\n'
            f'SCRIPT_DIR="{script_dir}"\n'
            f'cd "{str(tmp_path)}"\n'
            f'{chain_block}\n'
        )

        result = subprocess.run(
            ['bash', '-c', test_script],
            env=workspace["env"],
            capture_output=True,
            text=True,
        )

        # Verify sbatch was called
        sbatch_log = workspace["sbatch_log"]
        assert sbatch_log.exists(), (
            f"sbatch was never called. "
            f"Script stderr: {result.stderr}\n"
            f"Script stdout: {result.stdout}\n"
            f"Chain block:\n{chain_block}"
        )

        log_content = sbatch_log.read_text()
        assert "resume_0.sh" in log_content, (
            f"sbatch was called but not with resume_0.sh. Log: {log_content}"
        )

    def test_no_resume_when_no_resume_files(self, workspace):
        """
        When no .resume files exist, the chaining block should NOT call sbatch.
        """
        tmp_path = workspace["tmp_path"]
        script_dir = self._generate_scripts(tmp_path)

        main_path = os.path.join(script_dir, "main_0.sh")
        with open(main_path) as f:
            main_content = f.read()

        # Do NOT create any .resume files

        chain_block = self._extract_chaining_block(main_content)
        test_script = (
            f'#!/bin/bash\n'
            f'SCRIPT_DIR="{script_dir}"\n'
            f'cd "{str(tmp_path)}"\n'
            f'{chain_block}\n'
        )

        subprocess.run(
            ['bash', '-c', test_script],
            env=workspace["env"],
            capture_output=True,
            text=True,
        )

        # Verify sbatch was NOT called
        sbatch_log = workspace["sbatch_log"]
        assert not sbatch_log.exists(), (
            f"sbatch was called when no .resume files exist. "
            f"Log: {sbatch_log.read_text()}"
        )

    def test_resume_file_from_different_job_does_not_trigger(self, workspace):
        """
        A .resume file from a different SLURM_JOB_ID should not trigger chaining.
        """
        tmp_path = workspace["tmp_path"]
        script_dir = self._generate_scripts(tmp_path)

        main_path = os.path.join(script_dir, "main_0.sh")
        with open(main_path) as f:
            main_content = f.read()

        # Create a .resume file with a DIFFERENT slurm ID
        aslurm_dir = os.path.join(str(tmp_path), ".aslurm")
        wrong_id = "99999"
        assert wrong_id != self.FAKE_SLURM_ID
        resume_file = os.path.join(aslurm_dir, f"{wrong_id}_0.resume")
        with open(resume_file, 'w') as f:
            f.write("echo wrong_job")

        chain_block = self._extract_chaining_block(main_content)
        test_script = (
            f'#!/bin/bash\n'
            f'SCRIPT_DIR="{script_dir}"\n'
            f'cd "{str(tmp_path)}"\n'
            f'{chain_block}\n'
        )

        subprocess.run(
            ['bash', '-c', test_script],
            env=workspace["env"],
            capture_output=True,
            text=True,
        )

        sbatch_log = workspace["sbatch_log"]
        assert not sbatch_log.exists(), (
            f"sbatch was triggered by a .resume file from a different job ID. "
            f"Log: {sbatch_log.read_text()}"
        )

    def test_resume_script_chaining_block_also_self_chains(self, workspace):
        """
        The resume script itself should also have a chaining block that can
        trigger another round of resumption.
        """
        tmp_path = workspace["tmp_path"]
        script_dir = self._generate_scripts(tmp_path)

        resume_path = os.path.join(script_dir, "resume_0.sh")
        assert os.path.exists(resume_path), "resume_0.sh not found"
        with open(resume_path) as f:
            resume_content = f.read()

        # Create a .resume file
        aslurm_dir = os.path.join(str(tmp_path), ".aslurm")
        resume_file = os.path.join(aslurm_dir, f"{self.FAKE_SLURM_ID}_0.resume")
        with open(resume_file, 'w') as f:
            f.write("echo resumed_again")

        chain_block = self._extract_chaining_block(resume_content)
        test_script = (
            f'#!/bin/bash\n'
            f'SCRIPT_DIR="{script_dir}"\n'
            f'cd "{str(tmp_path)}"\n'
            f'{chain_block}\n'
        )

        result = subprocess.run(
            ['bash', '-c', test_script],
            env=workspace["env"],
            capture_output=True,
            text=True,
        )

        sbatch_log = workspace["sbatch_log"]
        assert sbatch_log.exists(), (
            f"Resume script's chaining block did not trigger sbatch. "
            f"stderr: {result.stderr}"
        )

        log_content = sbatch_log.read_text()
        assert "resume_0.sh" in log_content, (
            f"Resume script should chain to itself. Log: {log_content}"
        )

    def test_multi_task_resume_files(self, workspace):
        """
        When multiple tasks create .resume files, the chaining should still trigger.
        This tests the glob pattern matching multiple .resume files.
        """
        tmp_path = workspace["tmp_path"]

        # Generate scripts with multiple commands
        runner = CliRunner()
        result = runner.invoke(aslurm, [
            f'--archive-path={str(tmp_path)}',
            '--config-name=haicore_1gpu',
            '-d',
            '-mt', '3',
            'cmd', 'echo task0',
            'cmd', 'echo task1',
            'cmd', 'echo task2',
        ])
        assert result.exit_code == 0, f"Script generation failed: {result.output}"

        # Find script dir
        aslurm_path = tmp_path / ".aslurm"
        script_dirs = [
            os.path.join(str(aslurm_path), d)
            for d in os.listdir(str(aslurm_path))
            if os.path.isdir(os.path.join(str(aslurm_path), d))
        ]
        script_dir = script_dirs[0]

        main_path = os.path.join(script_dir, "main_0.sh")
        with open(main_path) as f:
            main_content = f.read()

        # Create .resume files for task 0 and task 2 (but not task 1 — partial resume)
        aslurm_dir = str(aslurm_path)
        for task_idx in [0, 2]:
            resume_file = os.path.join(aslurm_dir, f"{self.FAKE_SLURM_ID}_{task_idx}.resume")
            with open(resume_file, 'w') as f:
                f.write(f"echo resumed_task_{task_idx}")

        chain_block = self._extract_chaining_block(main_content)
        test_script = (
            f'#!/bin/bash\n'
            f'SCRIPT_DIR="{script_dir}"\n'
            f'cd "{str(tmp_path)}"\n'
            f'{chain_block}\n'
        )

        subprocess.run(
            ['bash', '-c', test_script],
            env=workspace["env"],
            capture_output=True,
            text=True,
        )

        sbatch_log = workspace["sbatch_log"]
        assert sbatch_log.exists(), "sbatch should be triggered when any .resume files exist"
