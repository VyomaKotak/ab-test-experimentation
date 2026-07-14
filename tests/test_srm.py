from src.srm import srm_check


def test_clean_even_split_passes():
    result = srm_check({"control": 50_012, "treatment": 49_988},
                       {"control": 0.5, "treatment": 0.5})
    assert result["passed"] is True


def test_broken_split_fails():
    result = srm_check({"control": 47_000, "treatment": 53_000},
                       {"control": 0.5, "treatment": 0.5})
    assert result["passed"] is False


def test_intentional_imbalance_is_not_an_srm():
    # Criteo's 85 / 15 split is by design. Testing it against 50 / 50 would be
    # a false alarm, and this test exists to lock in that we never do that.
    result = srm_check({"control": 150_100, "treatment": 849_900},
                       {"control": 0.15, "treatment": 0.85})
    assert result["passed"] is True


def test_intentional_imbalance_measured_against_wrong_ratio_fails():
    result = srm_check({"control": 150_100, "treatment": 849_900},
                       {"control": 0.5, "treatment": 0.5})
    assert result["passed"] is False