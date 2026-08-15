"""Domain-agnostic discrete event simulation core."""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Event:
    time: float
    priority: int = 0
    kind: str = ""
    payload: Any = None


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[tuple[float, int, int, Event]] = []
        self._seq = 0

    def push(self, e: Event) -> None:
        heapq.heappush(self._heap, (e.time, e.priority, self._seq, e))
        self._seq += 1

    def pop(self) -> Event:
        return heapq.heappop(self._heap)[3]

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def advance(self, t: float) -> None:
        self.now = t


class Simulation:
    def __init__(self) -> None:
        self.clock = Clock()
        self.queue = EventQueue()
        self._handlers: dict[str, Callable[[Event, "Simulation"], None]] = {}

    def register(self, kind: str, handler: Callable[[Event, "Simulation"], None]) -> None:
        self._handlers[kind] = handler

    def schedule(self, time: float, kind: str, payload: Any = None, priority: int = 0) -> None:
        self.queue.push(Event(time=time, kind=kind, payload=payload, priority=priority))

    def run(self, until: float | None = None) -> None:
        while self.queue:
            e = self.queue.pop()
            if until is not None and e.time > until:
                self.clock.advance(until)
                break
            self.clock.advance(e.time)
            handler = self._handlers.get(e.kind)
            if handler is not None:
                handler(e, self)
