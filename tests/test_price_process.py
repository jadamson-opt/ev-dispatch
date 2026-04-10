import numpy as np

from ev_dispatch.price_process import PriceProcess, PriceProcessConfig


def make_process(seed: int = 42) -> PriceProcess:
    return PriceProcess(PriceProcessConfig(), seed=seed)


class TestSampleScenario:
    def test_output_shape(self):
        process = make_process()
        prices = process.sample_scenario()
        assert prices.shape == (48,)

    def test_prices_non_negative(self):
        process = make_process()
        prices = process.sample_scenario()
        assert np.all(prices >= 0)

    def test_same_seed_produces_same_scenario(self):
        prices_a = make_process(seed=42).sample_scenario()
        prices_b = make_process(seed=42).sample_scenario()
        np.testing.assert_array_equal(prices_a, prices_b)

    def test_different_seeds_produce_different_scenarios(self):
        prices_a = make_process(seed=42).sample_scenario()
        prices_b = make_process(seed=99).sample_scenario()
        assert not np.array_equal(prices_a, prices_b)

    def test_sample_scenarios_shape(self):
        process = make_process()
        scenarios = process.sample_scenarios(10)
        assert scenarios.shape == (10, 48)

    def test_evening_peak_higher_than_overnight(self):
        # Average over many scenarios to smooth out noise
        process = make_process()
        scenarios = process.sample_scenarios(200)
        evening_peak = scenarios[:, 0:8].mean()  # periods 0-8: 4pm-8pm
        overnight = scenarios[:, 16:28].mean()  # periods 16-28: midnight-6am
        assert evening_peak > overnight
