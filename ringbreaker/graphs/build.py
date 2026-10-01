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
from typing import Any, Iterable, Iterator

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
        self._incoming: dict[str, list[Payment]] = {}
        self._outgoing: dict[str, list[Payment]] = {}
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

    def get_signup_at(
        self, account_id: str, as_of: datetime | str | None = None
    ) -> datetime | None:
        signup = self._signup_at.get(str(account_id))
        if signup is None:
            return None
        if as_of is not None:
            cutoff = _as_of_or_max(as_of)
            if cutoff is not None and signup > cutoff:
                return None
        return signup

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
        self._incoming.setdefault(receiver, []).append(payment)
        self._outgoing.setdefault(sender, []).append(payment)
        return payment

    def payments(
        self,
        as_of: datetime | str | None = None,
        exclude_transaction_id: str | None = None,
        since: datetime | str | None = None,
    ) -> list[Payment]:
        cutoff = _as_of_or_max(as_of)
        since_dt = _as_of_or_max(since)
        out: list[Payment] = []
        for payment in self._payments:
            if not _visible(payment.timestamp, cutoff):
                continue
            if since_dt is not None and payment.timestamp < since_dt:
                continue
            if (
                exclude_transaction_id is not None
                and payment.transaction_id == exclude_transaction_id
            ):
                continue
            out.append(payment)
        return out

    def windowed(
        self,
        since: datetime | str | None = None,
        as_of: datetime | str | None = None,
    ) -> TransactionGraph:
        """Create a lightweight TransactionGraph containing only payments between since and as_of."""
        sub = TransactionGraph()
        # Copy signup timestamps
        sub._signup_at = dict(self._signup_at)
        sub_payments = self.payments(as_of=as_of, since=since)
        for p in sub_payments:
            sub.add_payment(
                p.sender,
                p.receiver,
                p.amount,
                p.timestamp,
                transaction_id=p.transaction_id,
                **p.extra,
            )
        return sub

    def get_out_neighbors(
        self,
        account_id: str,
        as_of: datetime | str | None = None,
        exclude_transaction_id: str | None = None,
    ) -> list[str]:
        account_id = str(account_id)
        seen: list[str] = []
        found: set[str] = set()
        for payment in self.outgoing_payments(
            account_id, as_of=as_of, exclude_transaction_id=exclude_transaction_id
        ):
            if payment.receiver not in found:
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
        for payment in self.incoming_payments(
            account_id, as_of=as_of, exclude_transaction_id=exclude_transaction_id
        ):
            if payment.sender not in found:
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
        cutoff = _as_of_or_max(as_of)
        return [
            p
            for p in self._outgoing.get(sender, [])
            if p.receiver == receiver
            and _visible(p.timestamp, cutoff)
            and (exclude_transaction_id is None or p.transaction_id != exclude_transaction_id)
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
        cutoff = _as_of_or_max(as_of)
        res = [
            p
            for p in self._outgoing.get(a, [])
            if p.receiver == b
            and _visible(p.timestamp, cutoff)
            and (exclude_transaction_id is None or p.transaction_id != exclude_transaction_id)
        ] + [
            p
            for p in self._outgoing.get(b, [])
            if p.receiver == a
            and _visible(p.timestamp, cutoff)
            and (exclude_transaction_id is None or p.transaction_id != exclude_transaction_id)
        ]
        res.sort(key=lambda p: p.timestamp)
        return res

    def get_account_history(
        self,
        account_id: str,
        as_of: datetime | str | None = None,
        exclude_transaction_id: str | None = None,
    ) -> list[Payment]:
        account_id = str(account_id)
        inc = self.incoming_payments(
            account_id, as_of=as_of, exclude_transaction_id=exclude_transaction_id
        )
        out = self.outgoing_payments(
            account_id, as_of=as_of, exclude_transaction_id=exclude_transaction_id
        )
        merged = inc + out
        merged.sort(key=lambda p: p.timestamp)
        return merged

    def incoming_payments(
        self,
        account_id: str,
        as_of: datetime | str | None = None,
        exclude_transaction_id: str | None = None,
    ) -> list[Payment]:
        account_id = str(account_id)
        cutoff = _as_of_or_max(as_of)
        return [
            p
            for p in self._incoming.get(account_id, [])
            if _visible(p.timestamp, cutoff)
            and (exclude_transaction_id is None or p.transaction_id != exclude_transaction_id)
        ]

    def outgoing_payments(
        self,
        account_id: str,
        as_of: datetime | str | None = None,
        exclude_transaction_id: str | None = None,
    ) -> list[Payment]:
        account_id = str(account_id)
        cutoff = _as_of_or_max(as_of)
        return [
            p
            for p in self._outgoing.get(account_id, [])
            if _visible(p.timestamp, cutoff)
            and (exclude_transaction_id is None or p.transaction_id != exclude_transaction_id)
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

    def account_ids(self, as_of: datetime | str | None = None) -> list[str]:
        all_accounts = [n for n, data in self.graph.nodes(data=True) if data.get("kind") == "account"]
        if as_of is None:
            return all_accounts
        cutoff = _as_of_or_max(as_of)
        visible: list[str] = []
        for acc in all_accounts:
            signup = self.get_signup_at(acc, as_of=cutoff)
            if signup is not None:
                visible.append(acc)
            elif self.incoming_payments(acc, as_of=cutoff) or self.outgoing_payments(acc, as_of=cutoff):
                visible.append(acc)
        return visible


class IdentityGraph:
    """Bipartite account -- identity-fragment graph (F4).

    Identity types are generic: ``device``, ``phone``, ``ip``, ``address``.
    Unknown extra types are allowed so Person 3 can add fields later.
    """

    def __init__(self) -> None:
        self.graph = nx.Graph()
        self._links: list[tuple[str, str, str, datetime]] = []
        # Indexes: one entry per (account, type, value), keeping the earliest
        # observation time, so as-of lookups do not scan every link.
        self._first_seen: dict[tuple[str, str, str], datetime] = {}
        self._by_account: dict[str, list[tuple[str, str]]] = {}
        self._by_identity: dict[tuple[str, str], list[str]] = {}

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
        key = (account_id, identity_type, identity_value)
        known = self._first_seen.get(key)
        if known is not None:
            if ts < known:
                self._first_seen[key] = ts
                self._links.append((account_id, identity_type, identity_value, ts))
            return
        self._first_seen[key] = ts
        self._by_account.setdefault(account_id, []).append((identity_type, identity_value))
        self._by_identity.setdefault((identity_type, identity_value), []).append(account_id)
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
        cutoff = _as_of_or_max(as_of)
        out: list[dict[str, str]] = []
        for itype, ivalue in self._by_account.get(account_id, []):
            if wanted is not None and itype != wanted:
                continue
            if not _visible(self._first_seen[(account_id, itype, ivalue)], cutoff):
                continue
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
        cutoff = _as_of_or_max(as_of)
        return [
            acc
            for acc in self._by_identity.get((identity_type, identity_value), [])
            if _visible(self._first_seen[(acc, identity_type, identity_value)], cutoff)
        ]

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

    def get_shared_phone_cluster(
        self,
        phone: str,
        as_of: datetime | str | None = None,
    ) -> list[str]:
        return self.get_accounts_for_identity("phone", phone, as_of=as_of)

    def get_shared_ip_cluster(
        self,
        ip: str,
        as_of: datetime | str | None = None,
    ) -> list[str]:
        return self.get_accounts_for_identity("ip", ip, as_of=as_of)

    def get_shared_address_cluster(
        self,
        address: str,
        as_of: datetime | str | None = None,
    ) -> list[str]:
        return self.get_accounts_for_identity("address", address, as_of=as_of)

    def as_of_graph(self, as_of: datetime | str | None = None) -> nx.Graph:
        """Build NetworkX bipartite Graph of visible accounts and identity fragments."""
        graph = nx.Graph()
        for acc, itype, ivalue in self._visible_links(as_of=as_of):
            ident_id = identity_node_id(itype, ivalue)
            graph.add_node(acc, kind="account")
            graph.add_node(
                ident_id,
                kind="identity",
                identity_type=itype,
                identity_value=ivalue,
            )
            graph.add_edge(acc, ident_id, identity_type=itype)
        return graph

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
