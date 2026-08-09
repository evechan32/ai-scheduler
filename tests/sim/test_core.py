# tests/sim/test_core.py
from moesim.sim.core import Event, EventQueue, Clock, Simulation


def test_event_is_frozen_dataclass():
    e = Event(time=1.5, kind="load")
    assert e.time == 1.5 and e.kind == "load" and e.priority == 0
    try:
        e.time = 2.0
        assert False, "Event must be immutable"
    except Exception:
        pass


def test_event_queue_ordered_pop():
    q = EventQueue()
    q.push(Event(time=3.0))
    q.push(Event(time=1.0))
    q.push(Event(time=2.0))
    assert len(q) == 3
    assert q.pop().time == 1.0
    assert q.pop().time == 2.0
    assert q.pop().time == 3.0
    assert not q


def test_clock_advance():
    c = Clock()
    assert c.now == 0.0
    c.advance(2.5)
    assert c.now == 2.5


def test_simulation_fires_handlers_in_time_order():
    sim = Simulation()
    order = []
    sim.register("a", lambda e, s: order.append(("a", e.time)))
    sim.register("b", lambda e, s: order.append(("b", e.time)))
    sim.schedule(2.0, "b")
    sim.schedule(1.0, "a")
    sim.run()
    assert order == [("a", 1.0), ("b", 2.0)]
    assert sim.clock.now == 2.0


def test_simulation_until_stops_early():
    sim = Simulation()
    fired = []
    sim.register("tick", lambda e, s: fired.append(e.time))
    for t in (1.0, 2.0, 3.0):
        sim.schedule(t, "tick")
    sim.run(until=2.5)
    assert fired == [1.0, 2.0]
    assert sim.clock.now == 2.5


def test_handler_can_schedule_new_events():
    sim = Simulation()
    seen = []

    def spawn(e, s):
        seen.append(e.time)
        if e.time < 3.0:
            s.schedule(e.time + 1.0, "spawn")

    sim.register("spawn", spawn)
    sim.schedule(0.0, "spawn")
    sim.run()
    assert seen == [0.0, 1.0, 2.0, 3.0]
