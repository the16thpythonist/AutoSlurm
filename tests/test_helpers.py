import os
import pytest

from .utils import ASSETS_PATH, ARTIFACTS_PATH
from auto_slurm.helpers import TEMPLATE_ENV
from auto_slurm.helpers import create_slurm_jobs, Batched
from auto_slurm.helpers import split_top_level_commas, expand_commands


def test_saving_artifacts():
    file_path = os.path.join(ARTIFACTS_PATH, "test_artifact.txt")
    with open(file_path, "w") as f:
        f.write("This is a test artifact.")
    
    assert os.path.exists(file_path), "Artifact file was not created."


def test_create_slurm_jobs_basic():
    # Use actual templates from the templates directory with absolute path loader
    main_template = TEMPLATE_ENV.get_template("main.sh.j2")
    resume_template = TEMPLATE_ENV.get_template("resume.sh.j2")

    fillers = {"user": "test"}
    commands = ["echo 1", "echo 2"]
    options = {"gpus_per_task": 2}

    main_script, resume_script = create_slurm_jobs(
        fillers=fillers,
        commands=commands,
        options=options,
        main_template=main_template,
        resume_template=resume_template,
    )

    # Save artifacts
    with open(os.path.join(ARTIFACTS_PATH, "main_basic.sh"), "w") as f:
        f.write(main_script)
    with open(os.path.join(ARTIFACTS_PATH, "resume_basic.sh"), "w") as f:
        f.write(resume_script)

    # Check that commands are in the main script
    assert "echo 1" in main_script and "echo 2" in main_script
    # Resume script should use eval/cat, not the original commands
    assert "eval" in resume_script and "cat" in resume_script
    assert "_0.resume" in resume_script and "_1.resume" in resume_script


def test_create_slurm_jobs_no_gpus():
    main_template = TEMPLATE_ENV.get_template("main.sh.j2")
    resume_template = TEMPLATE_ENV.get_template("resume.sh.j2")

    fillers = {"user": "test"}
    commands = ["run something"]
    options = {}  # No GPUs specified

    main_script, resume_script = create_slurm_jobs(
        fillers=fillers,
        commands=commands,
        options=options,
        main_template=main_template,
        resume_template=resume_template,
    )

    # Save artifacts
    with open(os.path.join(ARTIFACTS_PATH, "main_no_gpus.sh"), "w") as f:
        f.write(main_script)
    with open(os.path.join(ARTIFACTS_PATH, "resume_no_gpus.sh"), "w") as f:
        f.write(resume_script)

    assert "run something" in main_script
    # Resume script should use eval/cat, not the original commands
    assert "eval" in resume_script and "cat" in resume_script


def test_create_slurm_jobs_empty_commands():
    main_template = TEMPLATE_ENV.get_template("main.sh.j2")
    resume_template = TEMPLATE_ENV.get_template("resume.sh.j2")

    fillers = {"user": "test"}
    commands = []
    options = {"gpus_per_task": 1}

    main_script, resume_script = create_slurm_jobs(
        fillers=fillers,
        commands=commands,
        options=options,
        main_template=main_template,
        resume_template=resume_template,
    )

    # Save artifacts
    with open(os.path.join(ARTIFACTS_PATH, "main_empty.sh"), "w") as f:
        f.write(main_script)
    with open(os.path.join(ARTIFACTS_PATH, "resume_empty.sh"), "w") as f:
        f.write(resume_script)

    # Check that the script is not empty and has a SLURM shebang
    assert main_script.strip() != ""
    assert main_script.strip().startswith("#!/bin/bash")
    assert resume_script.strip() != ""
    assert resume_script.strip().startswith("#!/bin/bash")


class TestBatched:
    """Test class for the Batched generator wrapper."""
    
    def test_batched_basic(self):
        """Test basic batching functionality."""
        data = list(range(10))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batched = Batched(data, batch_size=3)
        
        batches = list(batched)
        expected = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        
        assert batches == expected

    def test_batched_exact_division(self):
        """Test batching when total elements divide evenly by batch size."""
        data = list(range(6))  # [0, 1, 2, 3, 4, 5]
        batched = Batched(data, batch_size=2)
        
        batches = list(batched)
        expected = [[0, 1], [2, 3], [4, 5]]
        
        assert batches == expected

    def test_batched_empty_iterable(self):
        """Test batching with empty iterable."""
        data = []
        batched = Batched(data, batch_size=3)
        
        batches = list(batched)
        expected = []
        
        assert batches == expected

    def test_batched_single_batch(self):
        """Test when all elements fit in a single batch."""
        data = [1, 2, 3]
        batched = Batched(data, batch_size=5)
        
        batches = list(batched)
        expected = [[1, 2, 3]]
        
        assert batches == expected

    def test_batched_batch_size_one(self):
        """Test batching with batch size of 1."""
        data = ['a', 'b', 'c']
        batched = Batched(data, batch_size=1)
        
        batches = list(batched)
        expected = [['a'], ['b'], ['c']]
        
        assert batches == expected

    def test_batched_randomize_false(self):
        """Test that randomize=False preserves order."""
        data = list(range(10))
        batched = Batched(data, batch_size=3, randomize=False)
        
        batches = list(batched)
        # Flatten batches to check original order is preserved
        flattened = [item for batch in batches for item in batch]
        
        assert flattened == data

    def test_batched_randomize_true(self):
        """Test that randomize=True shuffles elements."""
        data = list(range(20))
        batched = Batched(data, batch_size=5, randomize=True)
        
        batches = list(batched)
        # Flatten batches to check that elements are shuffled
        flattened = [item for batch in batches for item in batch]
        
        # Should contain same elements but likely in different order
        assert sorted(flattened) == sorted(data)
        # With 20 elements, it's extremely unlikely they'd be in the same order after shuffling
        # But we can't guarantee this, so we'll just check that all elements are present

    def test_batched_original_not_mutated(self):
        """Test that the original iterable is not mutated when randomize=True."""
        data = [1, 2, 3, 4, 5]
        original_data = data.copy()
        
        batched = Batched(data, batch_size=2, randomize=True)
        list(batched)  # Consume the generator
        
        # Original data should be unchanged
        assert data == original_data

    def test_batched_with_strings(self):
        """Test batching with string elements."""
        data = ['apple', 'banana', 'cherry', 'date', 'elderberry']
        batched = Batched(data, batch_size=2)
        
        batches = list(batched)
        expected = [['apple', 'banana'], ['cherry', 'date'], ['elderberry']]
        
        assert batches == expected

    def test_batched_with_tuple(self):
        """Test batching with tuple input."""
        data = (1, 2, 3, 4, 5)
        batched = Batched(data, batch_size=2)
        
        batches = list(batched)
        expected = [[1, 2], [3, 4], [5]]
        
        assert batches == expected

    def test_batched_invalid_batch_size(self):
        """Test that invalid batch sizes raise ValueError."""
        data = [1, 2, 3]
        
        with pytest.raises(ValueError, match="batch_size must be greater than 0"):
            Batched(data, batch_size=0)
        
        with pytest.raises(ValueError, match="batch_size must be greater than 0"):
            Batched(data, batch_size=-1)

    def test_batched_reusable(self):
        """Test that Batched can be iterated multiple times."""
        data = [1, 2, 3, 4]
        batched = Batched(data, batch_size=2, randomize=False)
        
        # First iteration
        batches1 = list(batched)
        # Second iteration
        batches2 = list(batched)
        
        expected = [[1, 2], [3, 4]]
        assert batches1 == expected
        assert batches2 == expected

    def test_batched_generator_input(self):
        """Test batching with generator input."""
        def number_generator():
            for i in range(5):
                yield i

        batched = Batched(number_generator(), batch_size=2)

        batches = list(batched)
        expected = [[0, 1], [2, 3], [4]]

        assert batches == expected


class TestSplitTopLevelCommas:
    """Test class for the split_top_level_commas function."""

    def test_simple_split(self):
        """Test splitting a simple comma-separated string."""
        result = split_top_level_commas("a, b, c")
        assert result == ["a", "b", "c"]

    def test_no_commas(self):
        """Test string with no commas returns single element."""
        result = split_top_level_commas("single_value")
        assert result == ["single_value"]

    def test_empty_string(self):
        """Test empty string returns single empty element."""
        result = split_top_level_commas("")
        assert result == [""]

    def test_nested_brackets(self):
        """Test that commas inside brackets are not split."""
        result = split_top_level_commas("a, [b, c], d")
        assert result == ["a", "[b, c]", "d"]

    def test_nested_braces(self):
        """Test that commas inside braces are not split."""
        result = split_top_level_commas("x, {y, z}, w")
        assert result == ["x", "{y, z}", "w"]

    def test_nested_parentheses(self):
        """Test that commas inside parentheses are not split."""
        result = split_top_level_commas("func(a, b), other(c, d)")
        assert result == ["func(a, b)", "other(c, d)"]

    def test_deeply_nested(self):
        """Test deeply nested structures."""
        result = split_top_level_commas("a, [[b, c], d], e")
        assert result == ["a", "[[b, c], d]", "e"]

    def test_mixed_brackets(self):
        """Test mixed bracket types."""
        result = split_top_level_commas("a, [b, {c, d}], (e, f)")
        assert result == ["a", "[b, {c, d}]", "(e, f)"]

    def test_whitespace_handling(self):
        """Test that whitespace is properly stripped."""
        result = split_top_level_commas("  a  ,  b  ,  c  ")
        assert result == ["a", "b", "c"]

    def test_unbalanced_opening_bracket_raises(self):
        """Test that unbalanced opening bracket raises ValueError."""
        with pytest.raises(ValueError, match="Unbalanced brackets"):
            split_top_level_commas("a, [b, c")

    def test_unbalanced_closing_bracket_raises(self):
        """Test that unbalanced closing bracket raises ValueError."""
        with pytest.raises(ValueError, match="Unbalanced brackets"):
            split_top_level_commas("a, b], c")

    def test_complex_values(self):
        """Test splitting complex command-line style values."""
        result = split_top_level_commas("--lr=0.1, --config=[a,b,c], --name=test")
        assert result == ["--lr=0.1", "--config=[a,b,c]", "--name=test"]


class TestExpandCommands:
    """Test class for the expand_commands function."""

    # --- Paired List Expansion Tests (<[...]>) ---

    def test_paired_expansion_simple(self):
        """Test simple paired list expansion with two parameters."""
        commands = ["python train.py --lr=<[0.1, 0.01]> --bs=<[16, 32]>"]
        result = expand_commands(commands)
        expected = [
            "python train.py --lr=0.1 --bs=16",
            "python train.py --lr=0.01 --bs=32",
        ]
        assert result == expected

    def test_paired_expansion_single_param(self):
        """Test paired list expansion with single parameter."""
        commands = ["python train.py --lr=<[0.1, 0.01, 0.001]>"]
        result = expand_commands(commands)
        expected = [
            "python train.py --lr=0.1",
            "python train.py --lr=0.01",
            "python train.py --lr=0.001",
        ]
        assert result == expected

    def test_paired_expansion_three_params(self):
        """Test paired list expansion with three parameters."""
        commands = ["python train.py --a=<[1, 2]> --b=<[x, y]> --c=<[!, @]>"]
        result = expand_commands(commands)
        expected = [
            "python train.py --a=1 --b=x --c=!",
            "python train.py --a=2 --b=y --c=@",
        ]
        assert result == expected

    def test_paired_expansion_length_mismatch_raises(self):
        """Test that mismatched paired list lengths raise ValueError."""
        commands = ["python train.py --lr=<[0.1, 0.01, 0.001]> --bs=<[16, 32]>"]
        with pytest.raises(ValueError, match="Paired lists must have the same length"):
            expand_commands(commands)

    # --- Grid Search Expansion Tests (<{...}>) ---

    def test_grid_expansion_simple(self):
        """Test simple grid search expansion with two parameters."""
        commands = ["python train.py --lr=<{0.1, 0.01}> --bs=<{16, 32}>"]
        result = expand_commands(commands)
        expected = [
            "python train.py --lr=0.1 --bs=16",
            "python train.py --lr=0.1 --bs=32",
            "python train.py --lr=0.01 --bs=16",
            "python train.py --lr=0.01 --bs=32",
        ]
        assert result == expected

    def test_grid_expansion_single_param(self):
        """Test grid search expansion with single parameter."""
        commands = ["python train.py --lr=<{0.1, 0.01, 0.001}>"]
        result = expand_commands(commands)
        expected = [
            "python train.py --lr=0.1",
            "python train.py --lr=0.01",
            "python train.py --lr=0.001",
        ]
        assert result == expected

    def test_grid_expansion_three_params(self):
        """Test grid search expansion creates full cartesian product."""
        commands = ["python train.py --a=<{1, 2}> --b=<{x, y}> --c=<{!}>"]
        result = expand_commands(commands)
        # 2 * 2 * 1 = 4 combinations
        assert len(result) == 4
        assert "python train.py --a=1 --b=x --c=!" in result
        assert "python train.py --a=1 --b=y --c=!" in result
        assert "python train.py --a=2 --b=x --c=!" in result
        assert "python train.py --a=2 --b=y --c=!" in result

    def test_grid_expansion_asymmetric(self):
        """Test grid expansion with different sized parameter lists."""
        commands = ["python train.py --lr=<{0.1, 0.01, 0.001}> --bs=<{16, 32}>"]
        result = expand_commands(commands)
        # 3 * 2 = 6 combinations
        assert len(result) == 6

    # --- Mixed and Edge Cases ---

    def test_mixing_paired_and_grid_raises(self):
        """Test that mixing <[]> and <{}> in same command raises ValueError."""
        commands = ["python train.py --lr=<[0.1, 0.01]> --bs=<{16, 32}>"]
        with pytest.raises(ValueError, match="Cannot mix"):
            expand_commands(commands)

    def test_no_expansion_syntax(self):
        """Test that commands without expansion syntax are unchanged."""
        commands = ["python train.py --lr=0.1 --bs=16"]
        result = expand_commands(commands)
        assert result == commands

    def test_empty_commands_list(self):
        """Test that empty command list returns empty list."""
        result = expand_commands([])
        assert result == []

    def test_multiple_commands_mixed(self):
        """Test expanding multiple commands with different syntaxes."""
        commands = [
            "python train.py --lr=<{0.1, 0.01}>",
            "python eval.py --model=best",
            "python test.py --seed=<[1, 2, 3]>",
        ]
        result = expand_commands(commands)
        # First command: 2 expansions, Second: 1 (no expansion), Third: 3 expansions
        assert len(result) == 6
        assert "python train.py --lr=0.1" in result
        assert "python train.py --lr=0.01" in result
        assert "python eval.py --model=best" in result
        assert "python test.py --seed=1" in result
        assert "python test.py --seed=2" in result
        assert "python test.py --seed=3" in result

    def test_nested_brackets_in_values(self):
        """Test expansion with nested brackets in values."""
        commands = ["python train.py --config=<{[1,2], [3,4]}>"]
        result = expand_commands(commands)
        expected = [
            "python train.py --config=[1,2]",
            "python train.py --config=[3,4]",
        ]
        assert result == expected

    def test_whitespace_in_values(self):
        """Test that whitespace in values is properly handled."""
        commands = ["python train.py --name=<{ hello , world }>"]
        result = expand_commands(commands)
        expected = [
            "python train.py --name=hello",
            "python train.py --name=world",
        ]
        assert result == expected

    def test_special_characters_in_values(self):
        """Test expansion with special characters in values."""
        commands = ["python train.py --path=<{/path/to/a, /path/to/b}>"]
        result = expand_commands(commands)
        expected = [
            "python train.py --path=/path/to/a",
            "python train.py --path=/path/to/b",
        ]
        assert result == expected

    def test_single_value_expansion(self):
        """Test expansion with single value (no actual expansion)."""
        commands = ["python train.py --lr=<{0.1}>"]
        result = expand_commands(commands)
        expected = ["python train.py --lr=0.1"]
        assert result == expected

    def test_expansion_preserves_command_order(self):
        """Test that expansion preserves the order of commands."""
        commands = [
            "first_command",
            "python train.py --lr=<{0.1, 0.01}>",
            "last_command",
        ]
        result = expand_commands(commands)
        assert result[0] == "first_command"
        assert result[-1] == "last_command"
        # Middle should be the expanded commands
        assert result[1] == "python train.py --lr=0.1"
        assert result[2] == "python train.py --lr=0.01"


class TestResumeScriptRendering:
    """
    Layer 1: Template rendering tests for the resume/chain job feature.

    These tests verify that create_slurm_jobs() produces correct main and resume
    scripts. The resume script should:
    - NOT contain the original commands
    - Instead use eval/cat to read .resume files written by write_resume_file()
    - Preserve GPU assignment (CUDA_VISIBLE_DEVICES) and task indices
    - Self-chain: check for new .resume files and resubmit itself
    - Have the same SBATCH directives as the main script

    Expected interface change: create_slurm_jobs() gains a `resume_script_name`
    parameter (str) so the main template can reference the correct resume script
    and the resume template can reference itself for self-chaining.
    """

    def _render(self, commands, options=None, fillers=None, resume_script_name="resume_0.sh"):
        """Helper to render main + resume scripts with sensible defaults."""
        main_template = TEMPLATE_ENV.get_template("main.sh.j2")
        resume_template = TEMPLATE_ENV.get_template("resume.sh.j2")
        return create_slurm_jobs(
            fillers=fillers or {"job_name": "test_job"},
            commands=commands,
            options=options or {},
            main_template=main_template,
            resume_template=resume_template,
            resume_script_name=resume_script_name,
        )

    # --- Resume script: eval/cat commands ---

    def test_resume_script_reads_from_resume_files(self):
        """Resume script should use eval + cat to read commands from .resume files."""
        _, resume_script = self._render(
            commands=["python train.py --lr=0.01", "python train.py --lr=0.001"],
        )

        with open(os.path.join(ARTIFACTS_PATH, "resume_eval_cat.sh"), "w") as f:
            f.write(resume_script)

        assert "eval" in resume_script, "Resume script should use 'eval' to execute resume commands"
        assert "cat" in resume_script, "Resume script should use 'cat' to read .resume files"
        assert "PREVIOUS_SLURM_ID" in resume_script, \
            "Resume script should reference PREVIOUS_SLURM_ID"

    def test_resume_script_has_correct_task_indices(self):
        """Resume script should reference a .resume file for each task index."""
        _, resume_script = self._render(
            commands=["echo task0", "echo task1", "echo task2"],
        )

        for i in range(3):
            assert f"_{i}.resume" in resume_script, \
                f"Resume script should reference .resume file for task index {i}"

    def test_resume_script_does_not_contain_original_commands(self):
        """Resume script should NOT contain the original command strings."""
        _, resume_script = self._render(
            commands=["python train.py --very_unique_flag_12345"],
        )

        assert "very_unique_flag_12345" not in resume_script, \
            "Resume script should not contain original command text; it should use eval/cat"

    def test_resume_single_command(self):
        """Resume script should work correctly with a single command."""
        _, resume_script = self._render(commands=["echo hello"])

        assert "eval" in resume_script
        assert "_0.resume" in resume_script

    def test_resume_many_commands(self):
        """Resume script should handle many commands, each with its own .resume reference."""
        commands = [f"python run_{i}.py" for i in range(8)]
        _, resume_script = self._render(commands=commands)

        for i in range(8):
            assert f"_{i}.resume" in resume_script, \
                f"Missing .resume reference for task index {i}"

    # --- Resume script: self-chaining ---

    def test_resume_script_self_chains(self):
        """Resume script should check for new .resume files and resubmit itself."""
        _, resume_script = self._render(
            commands=["echo hello"],
            resume_script_name="resume_0.sh",
        )

        assert "compgen" in resume_script, \
            "Resume script should use compgen to check for .resume files"
        assert ".resume" in resume_script, \
            "Resume script should check for .resume file pattern"
        assert "sbatch" in resume_script, \
            "Resume script should call sbatch to resubmit"
        assert "resume_0.sh" in resume_script, \
            "Resume script should reference itself for chaining"

    def test_resume_script_self_chains_with_different_index(self):
        """Resume script for job index 2 should chain to resume_2.sh, not resume_0.sh."""
        _, resume_script = self._render(
            commands=["echo hello"],
            resume_script_name="resume_2.sh",
        )

        assert "resume_2.sh" in resume_script, \
            "resume_2.sh should reference itself, not resume_0.sh"

    # --- Main script: correct resume reference ---

    def test_main_script_references_correct_resume_filename(self):
        """Main script should reference the specific resume filename, not a hardcoded one."""
        main_0, _ = self._render(
            commands=["echo hello"],
            resume_script_name="resume_0.sh",
        )
        assert "resume_0.sh" in main_0

        main_1, _ = self._render(
            commands=["echo hello"],
            resume_script_name="resume_1.sh",
        )
        assert "resume_1.sh" in main_1

        main_3, _ = self._render(
            commands=["echo hello"],
            resume_script_name="resume_3.sh",
        )
        assert "resume_3.sh" in main_3

    # --- Resume script: GPU assignment ---

    def test_resume_script_has_gpu_assignment(self):
        """When gpus_per_task is set, resume script should assign CUDA_VISIBLE_DEVICES."""
        _, resume_script = self._render(
            commands=["python train1.py", "python train2.py"],
            options={"gpus_per_task": 1},
        )

        with open(os.path.join(ARTIFACTS_PATH, "resume_gpu.sh"), "w") as f:
            f.write(resume_script)

        assert "CUDA_VISIBLE_DEVICES=0" in resume_script, \
            "GPU 0 assignment not found in resume script"
        assert "CUDA_VISIBLE_DEVICES=1" in resume_script, \
            "GPU 1 assignment not found in resume script"

    def test_resume_script_no_gpu_when_not_configured(self):
        """When gpus_per_task is not set, resume script should not set CUDA_VISIBLE_DEVICES."""
        _, resume_script = self._render(
            commands=["python train1.py", "python train2.py"],
            options={},
        )

        assert "CUDA_VISIBLE_DEVICES" not in resume_script

    def test_resume_script_multi_gpu_per_task(self):
        """With gpus_per_task=2, each resumed task should get 2 GPUs."""
        _, resume_script = self._render(
            commands=["python train1.py", "python train2.py"],
            options={"gpus_per_task": 2},
        )

        # Task 0 gets GPUs 0,1 and task 1 gets GPUs 2,3
        assert "CUDA_VISIBLE_DEVICES=0,1" in resume_script
        assert "CUDA_VISIBLE_DEVICES=2,3" in resume_script

    # --- Resume script: SLURM_SUBMIT_TASK_INDEX ---

    def test_resume_script_has_task_index_env_var(self):
        """Resume script should set SLURM_SUBMIT_TASK_INDEX for each resumed task."""
        _, resume_script = self._render(
            commands=["echo task0", "echo task1"],
        )

        assert "SLURM_SUBMIT_TASK_INDEX=0" in resume_script
        assert "SLURM_SUBMIT_TASK_INDEX=1" in resume_script

    # --- Resume script: SBATCH directives ---

    def test_resume_script_has_matching_sbatch_directives(self):
        """Resume script should have the same SBATCH directives as the main script."""
        fillers = {
            "job_name": "test_job",
            "time": "02:00:00",
            "mem": "16G",
            "partition": "gpu",
            "gres": "gpu:4",
        }
        main_script, resume_script = self._render(
            commands=["echo hello"],
            fillers=fillers,
        )

        with open(os.path.join(ARTIFACTS_PATH, "resume_sbatch.sh"), "w") as f:
            f.write(resume_script)

        assert "#!/bin/bash" in resume_script
        assert "#SBATCH --time=02:00:00" in resume_script
        assert "#SBATCH --mem=16G" in resume_script
        assert "#SBATCH --partition=gpu" in resume_script
        assert "#SBATCH --gres=gpu:4" in resume_script


