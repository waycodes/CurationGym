"""Unit tests for policy schema and hashing."""

import pytest
from curationgym.policy.schema import Policy, policy_hash


class TestPolicy:
    def test_create_policy(self):
        policy = Policy(
            filters=[{"name": "length", "min_tokens": 50}],
            dedup={"method": "minhash", "threshold": 0.8},
            mixing={"domain:news": 0.3, "domain:wiki": 0.7},
        )
        assert len(policy.filters) == 1
        assert policy.dedup["method"] == "minhash"
        assert policy.mixing["domain:wiki"] == 0.7

    def test_policy_to_dict(self):
        policy = Policy(
            filters=[{"name": "quality", "threshold": 0.5}],
        )
        d = policy.to_dict()
        assert d["filters"][0]["name"] == "quality"

    def test_policy_from_dict(self):
        data = {
            "filters": [{"name": "lang", "lang": "en"}],
            "dedup": {"method": "exact"},
            "decontam": {"mode": "ngram", "ngram_size": 13},
        }
        policy = Policy.from_dict(data)
        assert policy.filters[0]["lang"] == "en"
        assert policy.decontam["mode"] == "ngram"

    def test_policy_hash_stable(self):
        policy1 = Policy(filters=[{"name": "a"}, {"name": "b"}])
        policy2 = Policy(filters=[{"name": "a"}, {"name": "b"}])
        assert policy_hash(policy1) == policy_hash(policy2)

    def test_policy_hash_differs(self):
        policy1 = Policy(filters=[{"name": "a"}])
        policy2 = Policy(filters=[{"name": "b"}])
        assert policy_hash(policy1) != policy_hash(policy2)

    def test_policy_hash_order_independent_mixing(self):
        policy1 = Policy(mixing={"a": 0.5, "b": 0.5})
        policy2 = Policy(mixing={"b": 0.5, "a": 0.5})
        assert policy_hash(policy1) == policy_hash(policy2)

    def test_policy_validate_mixing_weights(self):
        # Weights should sum to ~1
        policy = Policy(mixing={"a": 0.3, "b": 0.7})
        assert policy.validate_mixing_weights()

    def test_policy_empty(self):
        policy = Policy()
        assert policy.filters == []
        assert policy.mixing == {}
