import pytest  # noqa: F401

from aider.run_cmd import run_cmd


def test_run_cmd_echo():
    """
    Test the run_cmd function by executing a simple echo command.
    
    This test checks if the run_cmd function correctly executes the command
    'echo Hello, World!' and returns the expected exit code and output.
    """
    command = "echo Hello, World!"
    exit_code, output = run_cmd(command)

    assert exit_code == 0
    assert output.strip() == "Hello, World!"
