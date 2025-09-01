import os
import pytest
import jinja2 as j2
from io import StringIO

from .utils import ASSETS_PATH, ARTIFACTS_PATH
from auto_slurm.helpers import TEMPLATE_ENV
from auto_slurm.helpers import create_slurm_jobs, Batched, suppress_rich_click_output
import rich_click


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


