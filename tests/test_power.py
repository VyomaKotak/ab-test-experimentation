from src.power import required_sample_size, achieved_power


def test_smaller_effect_needs_more_users():
    big = required_sample_size(0.20, 0.10)
    small = required_sample_size(0.20, 0.02)
    assert small["n_total"] > big["n_total"]


def test_higher_power_needs_more_users():
    low = required_sample_size(0.20, 0.05, power=0.80)
    high = required_sample_size(0.20, 0.05, power=0.95)
    assert high["n_total"] > low["n_total"]


def test_rare_events_need_more_users():
    # Criteo conversion is 0.29 percent. That is the whole reason the sample is huge.
    common = required_sample_size(0.20, 0.10)
    rare = required_sample_size(0.0029, 0.10)
    assert rare["n_total"] > common["n_total"] * 50


def test_more_data_gives_more_power():
    low = achieved_power(10_000, 10_000, 0.20, 0.21)
    high = achieved_power(100_000, 100_000, 0.20, 0.21)
    assert high > low