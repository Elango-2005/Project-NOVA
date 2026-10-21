"""NOVA Layer 1 Event Bus Interface & In-Memory Implementation (ADR-002 / ADR-006 / NFR-08).

Implements the publish/subscribe event bus contract from NOVA_ADR.md:
- ADR-002: Internal components communicate through typed domain events.
- Engineering Principle 3 & NFR-08: 100% of agent steps, model invocations, and tool
  proposals emit structured events for streaming UX, debugging, and auditability.
- Phase 8 Final Recommendation: In-process pub/sub for single-device v1 operation behind
  a replaceable interface (`EventPublisherPort`).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from nova.core.domain.events import EventType, NovaEvent

EventSubscriber = Callable[[NovaEvent], None]


@runtime_checkable
class EventPublisherPort(Protocol):
    """Abstract interface for publishing and subscribing to structured NOVA events (ADR-002)."""

    def publish(self, event: NovaEvent) -> None:
        """Publish an immutable NovaEvent to all matching subscribers."""

    def subscribe(
        self,
        subscriber: EventSubscriber,
        event_type: Optional[EventType] = None,
    ) -> None:
        """Register a subscriber for a specific EventType or all events (when event_type is None)."""


class InMemoryEventBus:
    """Deterministic in-process event bus with an inspectable audit log (ADR-002 / NFR-08)."""

    def __init__(self) -> None:
        self._wildcard_subscribers: List[EventSubscriber] = []
        self._typed_subscribers: Dict[EventType, List[EventSubscriber]] = defaultdict(list)
        self._published_events: List[NovaEvent] = []

    def publish(self, event: NovaEvent) -> None:
        """Record the event in the audit log and dispatch synchronously to subscribers."""
        if not isinstance(event, NovaEvent):
            raise TypeError(f"Expected NovaEvent instance, got {type(event).__name__}.")
        self._published_events.append(event)

        for callback in list(self._typed_subscribers.get(event.event_type, ())):
            callback(event)
        for callback in list(self._wildcard_subscribers):
            callback(event)

    def subscribe(
        self,
        subscriber: EventSubscriber,
        event_type: Optional[EventType] = None,
    ) -> None:
        """Subscribe a callback to a specific EventType or all events."""
        if event_type is None:
            self._wildcard_subscribers.append(subscriber)
        else:
            self._typed_subscribers[event_type].append(subscriber)

    @property
    def published_events(self) -> Tuple[NovaEvent, ...]:
        """Return an immutable snapshot of all published events in chronological order."""
        return tuple(self._published_events)

    def events_by_type(self, event_type: EventType) -> Tuple[NovaEvent, ...]:
        """Filter published events by a specific EventType."""
        return tuple(e for e in self._published_events if e.event_type == event_type)

    # Alias for backwards compatibility
    events_By_type = events_by_type

    def clear(self) -> None:
        """Clear recorded events (for test isolation)."""
        self._published_events.clear()