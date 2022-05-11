from click.testing import CliRunner
import pytest

from src.capstone_project.train import train

@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()

def test_error_for_unsupported_alg(runner: CliRunner):
    with pytest.raises(ValueError):
        result = runner.invoke(
        train,
        [
            "-m",
            'reg',
        ])
        print(result)

def test_error_for_unsupported_tuning_method(runner: CliRunner):
    with pytest.raises(ValueError):
        result = runner.invoke(
        train,
        [
            "--tuning",
            'bayes',
        ])
        print(result)

def test_error_for_invalid_test_split_ratio(runner: CliRunner) -> None:
    result = runner.invoke(
        train,
        [
            "--test-split-ratio",
            42,
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--test-split-ratio'" in result.output