import sys
import unittest
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from part2_6_scenario_compliance import (  # noqa: E402
    build_scenario_compliance,
    parse_scenario_requirements,
)

PROMPTS_FILE = REPO_ROOT / "prompts" / "4scenarios.txt"


class ScenarioComplianceTests(unittest.TestCase):
    def test_parse_scenario_requirements_from_prompt_file(self) -> None:
        reqs = parse_scenario_requirements(PROMPTS_FILE)

        self.assertFalse(reqs["S1"]["expected_retraining"])
        self.assertTrue(reqs["S2"]["expected_retraining"])
        self.assertTrue(reqs["S2"]["expected_ground_truth"])
        self.assertTrue(reqs["S3"]["expected_multistep"])
        self.assertTrue(reqs["S4"]["expected_blockwise"])

    def test_build_scenario_compliance_reports_full_compliance_flags(self) -> None:
        protocol = pd.DataFrame(
            [
                {
                    "file_id": "run-1",
                    "scenario": "S2",
                    "t2_understands_retraining": True,
                    "t2_understands_groundtruth": True,
                    "t2_understands_blockwise": False,
                    "t3_retraining_implemented": True,
                    "t3_uses_ground_truth": True,
                    "t3_correct_block_size": False,
                    "t3_parse_success": True,
                }
            ]
        )
        reqs = parse_scenario_requirements(PROMPTS_FILE)
        out = build_scenario_compliance(protocol, reqs)

        self.assertEqual(out.loc[0, "semantic_compliance"], 100.0)
        self.assertEqual(out.loc[0, "implementation_compliance"], 100.0)
        self.assertTrue(out.loc[0, "fully_semantic_compliant"])
        self.assertTrue(out.loc[0, "fully_implementation_compliant"])

    def test_build_scenario_compliance_flags_partial_implementation_failures(self) -> None:
        protocol = pd.DataFrame(
            [
                {
                    "file_id": "run-2",
                    "scenario": "S2",
                    "t2_understands_retraining": True,
                    "t2_understands_groundtruth": True,
                    "t2_understands_blockwise": False,
                    "t3_retraining_implemented": True,
                    "t3_uses_ground_truth": False,
                    "t3_correct_block_size": False,
                    "t3_parse_success": True,
                }
            ]
        )
        reqs = parse_scenario_requirements(PROMPTS_FILE)
        out = build_scenario_compliance(protocol, reqs)

        self.assertEqual(out.loc[0, "semantic_compliance"], 100.0)
        self.assertEqual(out.loc[0, "implementation_compliance"], 50.0)
        self.assertTrue(out.loc[0, "fully_semantic_compliant"])
        self.assertFalse(out.loc[0, "fully_implementation_compliant"])


if __name__ == "__main__":
    unittest.main()
