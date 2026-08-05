"""
ARC-AGI-2 Test-Time Adaptation (TTA) Candidate Verification Engine
Executes model-generated Python transformation candidates against train prompt pairs.
"""

import sys
import io
import contextlib

class TestTimeAdaptationEvaluator:
    """Verifies candidate Python programs against prompt examples at inference time."""

    def __init__(self, timeout_sec=1.0):
        self.timeout_sec = timeout_sec

    def execute_candidate(self, code_str, input_grid):
        """Executes a generated code string with input_grid as argument."""
        namespace = {}
        try:
            exec(code_str, namespace)
            if 'transform' in namespace:
                result = namespace['transform'](input_grid)
                return result
        except Exception:
            return None
        return None

    def verify_candidate_on_task(self, code_str, task_data):
        """Returns True if the candidate program correctly transforms all train input grids into train output grids."""
        train_pairs = task_data.get("train", [])
        for pair in train_pairs:
            inp = pair["input"]
            expected = pair["output"]
            pred = self.execute_candidate(code_str, inp)
            if pred != expected:
                return False
        return True

if __name__ == "__main__":
    evaluator = TestTimeAdaptationEvaluator()
    sample_code = """
def transform(grid):
    return [row[::-1] for row in grid]
"""
    sample_task = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]}
        ]
    }
    is_valid = evaluator.verify_candidate_on_task(sample_code, sample_task)
    print(f"Sample Candidate Verification Result: {is_valid}")
