"""
src/tests/api/test_utils.py

Author: Jared Moore
Date: July, 2025

Tests for the api utils.
"""

import unittest
from collections import Counter
from datetime import timedelta

from api.utils import ServerSettings, min_positive_timedelta_diff
from experiment.condition import Condition, Roles


class TestUtils(unittest.TestCase):

    def test_min_positive_timedelta_diff(self):
        td1 = timedelta(seconds=5)
        td2 = timedelta(seconds=3)

        # 5 - 3 = 2
        diff = min_positive_timedelta_diff(td1, td2)
        self.assertEqual(diff, timedelta(seconds=2))

        # check symmetry with explicit argument names
        diff_rev = min_positive_timedelta_diff(td1=td2, td2=td1)
        self.assertEqual(diff_rev, timedelta(seconds=2))

        # identical timedeltas => zero
        diff_zero = min_positive_timedelta_diff(td1, td1)
        self.assertEqual(diff_zero, timedelta(0))

    def test_model_post_init_with_conditions(self):
        # prepare a small custom condition spec
        custom_roles = {"human_persuader": True, "llm_target": "gpt-4"}
        cond_list = [
            {"roles": custom_roles, "condition": {"factual_domain": False}, "count": 3},
            {"roles": custom_roles, "condition": {"factual_domain": False}, "count": 2},
        ]

        # bypass Pydantic’s __init__ / validation entirely
        settings = ServerSettings.model_construct()
        # manually assign what we need before model_post_init
        settings.conditions = cond_list
        settings.condition_num_rounds = Counter()

        # call the hook
        settings.model_post_init(None)

        # now verify it summed counts correctly
        roles_obj = Roles(**custom_roles)
        cond_key = Condition(roles=roles_obj, factual_domain=False)
        self.assertEqual(settings.condition_num_rounds, Counter({cond_key: 5}))


if __name__ == "__main__":
    unittest.main()
