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

    # Check that commands are in the output
    assert "echo 1" in main_script and "echo 2" in main_script
    assert "echo 1" in resume_script and "echo 2" in resume_script


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
    assert "run something" in resume_script


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


