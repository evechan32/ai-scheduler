def test_packages_importable():
    import moesim.sim
    import moesim.scheduler
    import moesim.executor
    assert moesim.sim.__name__ == "moesim.sim"
