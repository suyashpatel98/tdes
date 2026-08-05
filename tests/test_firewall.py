from __future__ import annotations

import unittest

from tdes.shards import FirewallError

from tests.support import RepositoryFixture


class FirewallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = RepositoryFixture()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fixture.close()

    def test_non_training_roles_cannot_train(self) -> None:
        for role in ("proxy", "validation", "eval"):
            manifest = next(item for item in self.fixture.repository.manifests if item["role"] == role)
            with self.subTest(role=role), self.assertRaises(FirewallError):
                self.fixture.repository.require_manifest_use(manifest["shard_id"], "train")

    def test_proxy_records_are_available_only_for_scoring(self) -> None:
        proxy = self.fixture.repository.records("score")
        train = self.fixture.repository.records("train")
        self.assertTrue(proxy)
        self.assertFalse({row["record_id"] for row in proxy} & {row["record_id"] for row in train})


if __name__ == "__main__":
    unittest.main()
