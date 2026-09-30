"""Transaction graph (F3) and identity-fragment graph (F4).

All queries that feed scoring accept an ``as_of`` timestamp. When ``as_of`` is
set, only payments and identity links with timestamp <= ``as_of`` are used.

Scoring a payment that has not yet moved: compute features with ``as_of`` equal
to the payment timestamp **before** calling :meth:`TransactionGraph.add_payment`,
or pass ``exclude_transaction_id`` so the candidate payment is ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Iterator, Optional

import networkx as nx

IDENTITY_TYPES = ("device", "phone", "ip", "address")


def parse_timestamp(value: Any) -> datetime:
    """Parse ISO strings, unix seconds, datetime, or pandas timestamps."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    if isinstance(value, (int, float)):
        return datetime.utcfromtimestamp(float(value))
    if isinstance(value, str):
        text = value.strip().replace("Z", "")
        if text.endswith("+00:00"):
            text = text[:-6]
        return datetime.fromisoformat(text)
    to_pydatetime = getattr(value, "to_pydatetime", None)
    if callable(to_pydatetime):
        converted = to_pydatetime()
        return converted.replace(tzinfo=None) if converted.tzinfo else converted
    raise TypeError(f"Unsupported timestamp type: {type(value)!r}")


def _as_of_or_max(as_of: datetime | str | None) -> datetime | None:
    if as_of is None:
        return None
    return parse_timestamp(as_of)


def _visible(ts: datetime, as_of: datetime | None) -> bool:
    if as_of is None:
        return True
    return ts <= as_of


@dataclass(frozen=True)
class Payment:
    """One directed payment. Extra simulator fields are stored in ``extra``."""

    sender: str
    receiver: str
    amount: float
    timestamp: datetime
    transaction_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "sender": self.sender,
            "receiver": self.receiver,
            "amount": self.amount,
            "timestamp": self.timestamp.isoformat(),
            "transaction_id": self.transaction_id,
        }
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


def identity_node_id(identity_type: str, identity_value: str) -> str:
    return f"{identity_type}:{identity_value}"


class TransactionGraph:
    """Directed multi-graph of accounts (nodes) and payments (edges).

    The full history is stored in insertion order, but every public query can
    be restricted with ``as_of`` so later payments cannot leak into earlier
    feature calculations.
    """

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self._payments: list[Payment] = []
        self._signup_at: dict[str, datetime] = {}
        self._edge_key = 0

    def register_account(
        self,
        account_id: str,
        signup_at: datetime | str | None = None,
        **attrs: Any,
    ) -> None:
        account_id = str(account_id)
        if account_id not in self.graph:
            self.graph.add_node(account_id, kind="account")
        if signup_at is not None:
            self._signup_at[account_id] = parse_timestamp(signup_at)
            self.graph.nodes[account_id]["signup_at"] = self._signup_at[account_id]
        for key, value in attrs.items():
            self.graph.nodes[account_id][key] = value

    def get_signup_at(self, account_id: str) -> datetime | None:
        return self._signup_at.get(str(account_id))

    def add_payment(
        self,
        sender: str,
        receiver: str,
        amount: float,
        timestamp: datetime | str,
        transaction_id: str | None = None,
        **extra: Any,
    ) -> Payment:
        """Append a directed payment A -> B. Graph updates immediately."""
        sender = str(sender)
        receiver = str(receiver)
        ts = parse_timestamp(timestamp)
        payment = Payment(
            sender=sender,
            receiver=receiver,
            amount=float(amount),
            timestamp=ts,
            transaction_id=None if transaction_id is None else str(transaction_id),
            extra=dict(extra),
        )
        self.register_account(sender)
        self.register_account(receiver)
        key = self._edge_key
        self._edge_key += 1
        self.graph.add_edge(
            sender,
            receiver,
            key=key,
            transaction_id=payment.transaction_id,
            timestamp=ts,
            amount=payment.amount,
            **extra,
        )
        self._payments.append(payment)
        return payment

    def payments(
        self,
        as_of: datetime | str | None = None,
        exclude_transaction_id: str | None = None,
    ) -> list[Payment]:
        cutoff = _as_of_or_max(as_of)
        out: list[Payment] = []
        for payment in self._payments:
            if not _visible(payment.timestamp, cutoff):
                continue
            if (
                exclude_transaction_id is not None
                and payment.transaction_id == exclude_transaction_id
            ):
                continue
            out.append(payment)
        return out

    def get_out_neighbors(
        self,
        account_id: str,
        as_of: datetime | str | None = None,
        exclude_transaction_id: str | None = None,
    ) -> list[str]:
        account_id = str(account_id)
        seen: list[str] = []
        found: set[str] = set()
        for payment in self.payments(as_of=as_of, exclude_transaction_id=exclude_transaction_id):
            if payment.sender == account_id and payment.receiver not in found:
                found.add(payment.receiver)
                seen.append(payment.receiver)
        return seen

    def get_in_neighbors(
        self,
        account_id: str,
        as_of: datetime | str | None = None,
        exclude_transaction_id: str | None = None,
    ) -> list[str]:
        account_id = str(account_id)
        seen: list[str] = []
        found: set[str] = set()
        for payment in self.payments(as_of=as_of, exclude_transaction_id=exclude_transaction_id):
            if payment.receiver == account_id and payment.sender not in found:
                found.add(payment.sender)
                seen.append(payment.sender)
        return seen

    def get_neighbors(
        self,
        account_id: str,
        as_of: datetime | str | None = None,
        exclude_transaction_id: str | None = None,
    ) -> list[str]:
        """Undirected counterparties (in or out)."""
        ordered: list[str] = []
        found: set[str] = set()
        for neighbor in self.get_out_neighbors(
            account_id, as_of=as_of, exclude_transaction_id=exclude_transaction_id
        ) + self.get_in_neighbors(
            account_id, as_of=as_of, exclude_transaction_id=exclude_transaction_id
        ):
            if neighbor not in found:
                found.add(neighbor)
                ordered.append(neighbor)
        return ordered

    def get_pair_history(
        self,
        sender: str,
        receiver: str,
        as_of: datetime | str | None = None,
        exclude_transaction_id: str | None = None,
    ) -> list[Payment]:
        sender = str(sender)
        receiver = str(receiver)
        return [
            p
            for p in self.payments(as_of=as_of, exclude_transaction_id=exclude_transaction_id)
            if p.sender == sender and p.receiver == receiver
        ]

    def get_transactions_between(
        self,
        account_a: str,
        account_b: str,
        as_of: datetime | str | None = None,
        exclude_transaction_id: str | None = None,
    ) -> list[Payment]:
        """Payments in either direction between two accounts."""
        a, b = str(account_a), str(account_b)
        return [
            p
            for p in self.payments(as_of=as_of, exclude_transaction_id=exclude_transaction_id)
            if {p.sender, p.receiver} == {a, b}
        ]

    def get_account_history(
        self,
        account_id: str,
        as_of: datetime | str | None = None,
        exclude_transaction_id: str | None = None,
    ) -> list[Payment]:
        account_id = str(account_id)
        return [
            p
            for p in self.payments(as_of=as_of, exclude_transaction_id=exclude_transaction_id)
            if p.sender == account_id or p.receiver == account_id
        ]

    def incoming_payments(
        self,
        account_id: str,
        as_of: datetime | str | None = None,
        exclude_transaction_id: str | None = None,
    ) -> list[Payment]:
        account_id = str(account_id)
        return [
            p
            for p in self.payments(as_of=as_of, exclude_transaction_id=exclude_transaction_id)
            if p.receiver == account_id
        ]

    def outgoing_payments(
        self,
        account_id: str,
        as_of: datetime | str | None = None,
        exclude_transaction_id: str | None = None,
    ) -> list[Payment]:
        account_id = str(account_id)
        return [
            p
            for p in self.payments(as_of=as_of, exclude_transaction_id=exclude_transaction_id)
            if p.sender == account_id
        ]

    def as_of_digraph(
        self,
        as_of: datetime | str | None = None,
        exclude_transaction_id: str | None = None,
        accounts: Iterable[str] | None = None,
    ) -> nx.MultiDiGraph:
        """Build a NetworkX snapshot containing only visible payments."""
        allowed = None if accounts is None else {str(a) for a in accounts}
        snapshot = nx.MultiDiGraph()
        for payment in self.payments(as_of=as_of, exclude_transaction_id=exclude_transaction_id):
            if allowed is not None and (
                payment.sender not in allowed or payment.receiver not in allowed
            ):
                continue
            snapshot.add_node(payment.sender, kind="account")
            snapshot.add_node(payment.receiver, kind="account")
            snapshot.add_edge(
                payment.sender,
                payment.receiver,
                transaction_id=payment.transaction_id,
                timestamp=payment.timestamp,
                amount=payment.amount,
                **payment.extra,
            )
        return snapshot

    def subgraph(
        self,
        account_id: str,
        hops: int = 2,
        as_of: datetime | str | None = None,
        exclude_transaction_id: str | None = None,
    ) -> nx.MultiDiGraph:
        """Bounded neighbourhood around an account on the undirected as-of graph."""
        account_id = str(account_id)
        if hops < 0:
            raise ValueError("hops must be >= 0")
        frontier = {account_id}
        visited = {account_id}
        for _ in range(hops):
            nxt: set[str] = set()
            for node in frontier:
                for neighbor in self.get_neighbors(
                    node, as_of=as_of, exclude_transaction_id=exclude_transaction_id
                ):
                    if neighbor not in visited:
                        nxt.add(neighbor)
                        visited.add(neighbor)
            frontier = nxt
            if not frontier:
                break
        return self.as_of_digraph(
            as_of=as_of,
            exclude_transaction_id=exclude_transaction_id,
            accounts=visited,
        )

    def snapshot(
        self,
        as_of: datetime | str | None = None,
        exclude_transaction_id: str | None = None,
    ) -> dict[str, Any]:
        """Serializable graph snapshot for Person 4 (dashboard / case file)."""
        graph = self.as_of_digraph(as_of=as_of, exclude_transaction_id=exclude_transaction_id)
        return serialize_digraph(graph)

    def account_ids(self) -> list[str]:
        return [n for n, data in self.graph.nodes(data=True) if data.get("kind") == "account"]


class IdentityGraph:
    """Bipartite account -- identity-fragment graph (F4).

    Identity types are generic: ``device``, ``phone``, ``ip``, ``address``.
    Unknown extra types are allowed so Person 3 can add fields later.
    """

    def __init__(self) -> None:
        self.graph = nx.Graph()
        self._links: list[tuple[str, str, str, datetime]] = []

    def add_account(self, account_id: str) -> None:
        account_id = str(account_id)
        if account_id not in self.graph:
            self.graph.add_node(account_id, kind="account")
        else:
            self.graph.nodes[account_id]["kind"] = "account"

    def add_identity_link(
        self,
        account_id: str,
        identity_type: str,
        identity_value: str,
        observed_at: datetime | str | None = None,
    ) -> None:
        if not identity_value:
            return
        account_id = str(account_id)
        identity_type = str(identity_type).lower().strip()
        identity_value = str(identity_value)
        ts = parse_timestamp(observed_at) if observed_at is not None else datetime.min
        ident_id = identity_node_id(identity_type, identity_value)
        self.add_account(account_id)
        self.graph.add_node(
            ident_id,
            kind="identity",
            identity_type=identity_type,
            identity_value=identity_value,
        )
        self.graph.add_edge(
            account_id,
            ident_id,
            observed_at=ts,
            identity_type=identity_type,
        )
        self._links.append((account_id, identity_type, identity_value, ts))

    def add_account_identities(
        self,
        account_id: str,
        *,
        device: str | None = None,
        phone: str | None = None,
        ip: str | None = None,
        address: str | None = None,
        observed_at: datetime | str | None = None,
        **extra: Any,
    ) -> None:
        """Attach optional identity fields. Missing/None fields are ignored."""
        mapping = {"device": device, "phone": phone, "ip": ip, "address": address}
        for key, value in extra.items():
            mapping[str(key)] = value
        for identity_type, identity_value in mapping.items():
            if identity_value in (None, ""):
                continue
            self.add_identity_link(
                account_id, identity_type, str(identity_value), observed_at=observed_at
            )

    def _visible_links(
        self, as_of: datetime | str | None = None
    ) -> Iterator[tuple[str, str, str]]:
        cutoff = _as_of_or_max(as_of)
        for account_id, identity_type, identity_value, ts in self._links:
            if _visible(ts, cutoff):
                yield account_id, identity_type, identity_value

    def get_identities(
        self,
        account_id: str,
        as_of: datetime | str | None = None,
        identity_type: str | None = None,
    ) -> list[dict[str, str]]:
        account_id = str(account_id)
        wanted = None if identity_type is None else identity_type.lower()
        out: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for acc, itype, ivalue in self._visible_links(as_of=as_of):
            if acc != account_id:
                continue
            if wanted is not None and itype != wanted:
                continue
            key = (itype, ivalue)
            if key in seen:
                continue
            seen.add(key)
            out.append({"identity_type": itype, "identity_value": ivalue})
        return out

    def get_accounts_for_identity(
        self,
        identity_type: str,
        identity_value: str,
        as_of: datetime | str | None = None,
    ) -> list[str]:
        identity_type = identity_type.lower()
        identity_value = str(identity_value)
        accounts: list[str] = []
        found: set[str] = set()
        for acc, itype, ivalue in self._visible_links(as_of=as_of):
            if itype == identity_type and ivalue == identity_value and acc not in found:
                found.add(acc)
                accounts.append(acc)
        return accounts

    def get_shared_identities(
        self,
        account_a: str,
        account_b: str,
        as_of: datetime | str | None = None,
    ) -> list[dict[str, str]]:
        a = {(i["identity_type"], i["identity_value"]) for i in self.get_identities(account_a, as_of=as_of)}
        b = {(i["identity_type"], i["identity_value"]) for i in self.get_identities(account_b, as_of=as_of)}
        return [
            {"identity_type": itype, "identity_value": ivalue}
            for itype, ivalue in sorted(a & b)
        ]

    def get_identity_neighbours(
        self,
        account_id: str,
        as_of: datetime | str | None = None,
        identity_type: str | None = None,
    ) -> list[str]:
        """Other accounts that share at least one identity fragment."""
        account_id = str(account_id)
        neighbours: list[str] = []
        found: set[str] = set()
        for ident in self.get_identities(account_id, as_of=as_of, identity_type=identity_type):
            for other in self.get_accounts_for_identity(
                ident["identity_type"], ident["identity_value"], as_of=as_of
            ):
                if other != account_id and other not in found:
                    found.add(other)
                    neighbours.append(other)
        return neighbours

    def get_shared_device_cluster(
        self,
        device_id: str,
        as_of: datetime | str | None = None,
    ) -> list[str]:
        return self.get_accounts_for_identity("device", device_id, as_of=as_of)

    def identity_clusters(
        self,
        identity_type: str = "device",
        min_accounts: int = 2,
        as_of: datetime | str | None = None,
    ) -> list[dict[str, Any]]:
        buckets: dict[str, list[str]] = {}
        for acc, itype, ivalue in self._visible_links(as_of=as_of):
            if itype != identity_type:
                continue
            buckets.setdefault(ivalue, [])
            if acc not in buckets[ivalue]:
                buckets[ivalue].append(acc)
        clusters = []
        for ivalue, members in buckets.items():
            if len(members) >= min_accounts:
                clusters.append(
                    {
                        "identity_type": identity_type,
                        "identity_value": ivalue,
                        "members": list(members),
                    }
                )
        return clusters

    def snapshot(self, as_of: datetime | str | None = None) -> dict[str, Any]:
        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []
        for acc, itype, ivalue in self._visible_links(as_of=as_of):
            ident_id = identity_node_id(itype, ivalue)
            nodes[acc] = {"id": acc, "kind": "account"}
            nodes[ident_id] = {
                "id": ident_id,
                "kind": "identity",
                "identity_type": itype,
                "identity_value": ivalue,
            }
            edges.append({"source": acc, "target": ident_id, "identity_type": itype})
        return {"nodes": list(nodes.values()), "edges": edges}


def serialize_digraph(graph: nx.MultiDiGraph) -> dict[str, Any]:
    nodes = [{"id": n, **{k: _jsonify(v) for k, v in data.items()}} for n, data in graph.nodes(data=True)]
    edges = []
    for u, v, data in graph.edges(data=True):
        item = {"source": u, "target": v}
        for key, value in data.items():
            item[key] = _jsonify(value)
        edges.append(item)
    return {"nodes": nodes, "edges": edges}


def _jsonify(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return value
