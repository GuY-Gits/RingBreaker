import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { Maximize2, Minus, Plus } from 'lucide-react';
import type { GraphEdge, GraphNode, GraphPayload } from '../../api/types';
import { ACTION_META, fmtTime, money, pct } from '../../lib/format';

type SimNode = GraphNode & d3.SimulationNodeDatum;
type SimLink = Omit<GraphEdge, 'source' | 'target'> & d3.SimulationLinkDatum<SimNode> & { source: string | SimNode; target: string | SimNode };

export interface NetworkGraphProps {
  data: GraphPayload;
  height?: number;
  selectedId?: string | null;
  /** Accounts to emphasise (e.g. members of a selected pattern); others fade. */
  emphasis?: Set<string> | null;
  onSelectNode?: (node: GraphNode | null) => void;
  onSelectEdge?: (edge: GraphEdge | null) => void;
  showIdentity?: boolean;
}

const nodeColor = (n: GraphNode) => {
  if (n.kind === 'identity') return 'var(--identity)';
  if (n.status === 'confirmed_fraud') return '#8f1d1d';
  if (n.risk >= 0.7) return 'var(--bad)';
  if (n.risk >= 0.3) return '#d59a1b';
  if (n.status === 'cleared') return 'var(--ok)';
  return '#8c8f96';
};

const edgeColor = (e: GraphEdge) => {
  if (e.kind === 'identity') return 'var(--identity)';
  if (e.is_trigger) return 'var(--accent)';
  const r = e.max_risk ?? 0;
  if (r >= 0.7) return 'var(--bad)';
  if (r >= 0.3) return '#d59a1b';
  return '#b8b4a8';
};

const nodeRadius = (n: GraphNode) =>
  n.kind === 'identity' ? 7 : 6 + Math.min(8, Math.sqrt(n.degree ?? 1)) + (n.highlight ? 3 : 0);

/** Stable positions per node id so live refreshes do not re-shuffle the layout. */
const positionCache = new Map<string, { x: number; y: number }>();

export function NetworkGraph({ data, height = 520, selectedId, emphasis, onSelectNode, onSelectEdge, showIdentity = true }: NetworkGraphProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown>>();
  const transformRef = useRef<d3.ZoomTransform | null>(null);
  const [tip, setTip] = useState<{ x: number; y: number; html: JSX.Element } | null>(null);
  const [width, setWidth] = useState(800);
  const handlers = useRef({ onSelectNode, onSelectEdge });
  handlers.current = { onSelectNode, onSelectEdge };

  const filtered = useMemo(() => {
    const nodes = data.nodes.filter((n) => showIdentity || n.kind !== 'identity');
    const ids = new Set(nodes.map((n) => n.id));
    const edges = data.edges.filter((e) => ids.has(e.source) && ids.has(e.target));
    return { nodes, edges };
  }, [data, showIdentity]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setWidth(Math.max(280, entry.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Build / rebuild the simulation when data or size changes.
  useEffect(() => {
    const svg = d3.select(svgRef.current!);
    svg.selectAll('*').remove();
    const w = width;
    const h = height;

    const defs = svg.append('defs');
    for (const [key, color] of [['n', '#b8b4a8'], ['w', '#d59a1b'], ['b', '#c62f2f'], ['t', '#e0492e']]) {
      defs.append('marker').attr('id', `arrow-${key}`).attr('viewBox', '0 -4 8 8').attr('refX', 8).attr('refY', 0)
        .attr('markerWidth', 7).attr('markerHeight', 7).attr('orient', 'auto')
        .append('path').attr('d', 'M0,-4L8,0L0,4').attr('fill', color);
    }

    const root = svg.append('g').attr('class', 'root');
    // Wheel zoom only with Ctrl/⌘ so scrolling the page over the graph still scrolls the page.
    const zoom = d3.zoom<SVGSVGElement, unknown>().scaleExtent([0.2, 4])
      .filter((ev: Event & { ctrlKey?: boolean; metaKey?: boolean; button?: number }) =>
        ev.type === 'wheel' ? Boolean(ev.ctrlKey || ev.metaKey) : !ev.button)
      .on('zoom', (ev) => {
      transformRef.current = ev.transform;
      root.attr('transform', ev.transform.toString());
      root.selectAll<SVGTextElement, SimNode>('text.node-label')
        .attr('display', (d) => (ev.transform.k > 1.15 || d.highlight || d.risk >= 0.3 || d.kind === 'identity' ? null : 'none'));
    });
    zoomRef.current = zoom;
    svg.call(zoom).on('dblclick.zoom', null);
    const keepView = transformRef.current;
    svg.on('click', () => {
      handlers.current.onSelectNode?.(null);
      handlers.current.onSelectEdge?.(null);
    });

    const nodes: SimNode[] = filtered.nodes.map((n) => {
      const cached = positionCache.get(n.id);
      return { ...n, x: cached?.x ?? w / 2 + (Math.random() - 0.5) * 80, y: cached?.y ?? h / 2 + (Math.random() - 0.5) * 80 };
    });
    const links: SimLink[] = filtered.edges.map((e) => ({ ...e }));

    const sim = d3.forceSimulation<SimNode>(nodes)
      .force('link', d3.forceLink<SimNode, SimLink>(links).id((d) => d.id)
        .distance((l) => (l.kind === 'identity' ? 48 : 95)).strength((l) => (l.kind === 'identity' ? 0.35 : 0.3)))
      .force('charge', d3.forceManyBody<SimNode>().strength((d) => (d.kind === 'identity' ? -140 : -320)).distanceMax(420))
      .force('collide', d3.forceCollide<SimNode>().radius((d) => nodeRadius(d) + 10))
      .force('x', d3.forceX(w / 2).strength(0.03))
      .force('y', d3.forceY(h / 2).strength(0.045));
    // Live refresh: if nearly every node already has a position, only settle gently.
    // Settle the layout synchronously (no dependence on animation frames), so
    // the graph appears already laid out; dragging restarts it interactively.
    const known = nodes.filter((n) => positionCache.has(n.id)).length;
    const reuse = nodes.length > 0 && known / nodes.length > 0.9;
    sim.stop();
    sim.alpha(reuse ? 0.15 : 1);
    for (let i = 0; i < 300 && sim.alpha() > sim.alphaMin(); i++) sim.tick();

    const link = root.append('g').selectAll<SVGLineElement, SimLink>('line').data(links).join('line')
      .attr('class', 'edge')
      .attr('stroke', (d) => edgeColor(d as unknown as GraphEdge))
      .attr('stroke-width', (d) => (d.kind === 'identity' ? 1.2 : Math.min(6, 1.2 + Math.log2(1 + (d.count ?? 1))) + (d.is_trigger ? 1.5 : 0)))
      .attr('stroke-dasharray', (d) => (d.kind === 'identity' ? '3 3' : null))
      .attr('stroke-opacity', 0.85)
      .attr('marker-end', (d) => {
        if (d.kind === 'identity') return null;
        if (d.is_trigger) return 'url(#arrow-t)';
        const r = d.max_risk ?? 0;
        return r >= 0.7 ? 'url(#arrow-b)' : r >= 0.3 ? 'url(#arrow-w)' : 'url(#arrow-n)';
      })
      .style('cursor', (d) => (d.kind === 'payment' ? 'pointer' : 'default'))
      .on('mouseenter', (ev, d) => {
        if (d.kind !== 'payment') return;
        const [x, y] = d3.pointer(ev, wrapRef.current);
        const s = typeof d.source === 'string' ? d.source : d.source.id;
        const t = typeof d.target === 'string' ? d.target : d.target.id;
        setTip({ x, y, html: (
          <>
            <div className="mono">{s} → {t}</div>
            <div>{d.count} payment{d.count === 1 ? '' : 's'} · {money(d.amount)}</div>
            <div>Last {fmtTime(d.last_ts)}</div>
            {d.max_risk != null && <div>Max risk {pct(d.max_risk)} · {ACTION_META[d.worst_action ?? 'ALLOW']?.label}</div>}
            {d.is_trigger && <div style={{ color: '#ffb4a3' }}>Flagged payment</div>}
          </>
        ) });
      })
      .on('mouseleave', () => setTip(null))
      .on('click', (ev, d) => {
        ev.stopPropagation();
        if (d.kind !== 'payment') return;
        const s = typeof d.source === 'string' ? d.source : d.source.id;
        const t = typeof d.target === 'string' ? d.target : d.target.id;
        handlers.current.onSelectEdge?.({ ...(d as unknown as GraphEdge), source: s, target: t });
      });

    const node = root.append('g').selectAll<SVGGElement, SimNode>('g').data(nodes, (d) => d.id).join('g')
      .attr('class', 'node')
      .style('cursor', 'pointer')
      .call(d3.drag<SVGGElement, SimNode>()
        .on('start', (ev, d) => { if (!ev.active) sim.alphaTarget(0.25).restart(); d.fx = d.x; d.fy = d.y; })
        .on('drag', (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
        .on('end', (ev, d) => { if (!ev.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }));

    node.each(function (d) {
      const g = d3.select(this);
      const r = nodeRadius(d);
      if (d.kind === 'identity') {
        g.append('rect').attr('x', -r).attr('y', -r).attr('width', r * 2).attr('height', r * 2)
          .attr('transform', 'rotate(45)').attr('fill', 'var(--identity-bg)').attr('stroke', 'var(--identity)').attr('stroke-width', 1.6);
      } else {
        if (d.highlight) g.append('circle').attr('r', r + 5).attr('fill', 'none').attr('stroke', 'var(--accent)').attr('stroke-width', 2);
        if (d.status === 'confirmed_fraud') g.append('circle').attr('r', r + 2.5).attr('fill', 'none').attr('stroke', '#8f1d1d').attr('stroke-width', 1.5).attr('stroke-dasharray', '2 2');
        g.append('circle').attr('class', 'body').attr('r', r).attr('fill', nodeColor(d)).attr('stroke', '#fff').attr('stroke-width', 1.8);
        if (d.patterns.length) g.append('circle').attr('r', 2.4).attr('cx', r * 0.72).attr('cy', -r * 0.72).attr('fill', 'var(--info)').attr('stroke', '#fff');
      }
      g.append('text').attr('class', 'node-label').attr('y', r + 12).attr('text-anchor', 'middle')
        .text(d.kind === 'identity' ? `${d.identity_type}` : d.label)
        .attr('display', d.highlight || d.risk >= 0.3 || d.kind === 'identity' ? null : 'none');
    });

    node
      .on('mouseenter', (ev, d) => {
        const [x, y] = d3.pointer(ev, wrapRef.current);
        setTip({ x, y, html: d.kind === 'identity' ? (
          <>
            <div className="mono">{d.identity_type}: {d.label}</div>
            <div>Shared by {d.shared_count} accounts in view</div>
          </>
        ) : (
          <>
            <div className="mono">{d.id}</div>
            <div>Risk {pct(d.risk)}{d.propagated_risk ? ` · propagated ${pct(d.propagated_risk)}` : ''}</div>
            {d.status === 'confirmed_fraud' && <div style={{ color: '#ffb4a3' }}>Confirmed fraud</div>}
            {d.patterns.length > 0 && <div>In {d.patterns.length} detected pattern{d.patterns.length > 1 ? 's' : ''}</div>}
            {d.lockstep && <div>Lockstep cluster member</div>}
            <div className="faint">{d.degree ?? 0} payments · click to inspect</div>
          </>
        ) });
      })
      .on('mouseleave', () => setTip(null))
      .on('click', (ev, d) => {
        ev.stopPropagation();
        handlers.current.onSelectNode?.(d);
      });

    function render() {
      link
        .attr('x1', (d) => (d.source as SimNode).x!)
        .attr('y1', (d) => (d.source as SimNode).y!)
        .attr('x2', (d) => {
          const s = d.source as SimNode, t = d.target as SimNode;
          const dx = t.x! - s.x!, dy = t.y! - s.y!, len = Math.hypot(dx, dy) || 1;
          return t.x! - (dx / len) * (nodeRadius(t) + (d.kind === 'identity' ? 0 : 3));
        })
        .attr('y2', (d) => {
          const s = d.source as SimNode, t = d.target as SimNode;
          const dx = t.x! - s.x!, dy = t.y! - s.y!, len = Math.hypot(dx, dy) || 1;
          return t.y! - (dy / len) * (nodeRadius(t) + (d.kind === 'identity' ? 0 : 3));
        });
      node.attr('transform', (d) => `translate(${d.x},${d.y})`);
    }
    sim.on('tick', render);
    sim.on('end', () => {
      nodes.forEach((n) => positionCache.set(n.id, { x: n.x!, y: n.y! }));
    });

    function fit(animate = true) {
      if (!nodes.length) return;
      const xs = nodes.map((n) => n.x!), ys = nodes.map((n) => n.y!);
      const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
      const k = Math.min(2.2, 0.9 / Math.max((x1 - x0 + 60) / w, (y1 - y0 + 60) / h));
      const t = d3.zoomIdentity.translate(w / 2, h / 2).scale(k).translate(-(x0 + x1) / 2, -(y0 + y1) / 2);
      (animate ? svg.transition().duration(400) : svg).call(zoom.transform as never, t);
    }
    (svgRef.current as unknown as { __fit?: () => void }).__fit = () => fit(true);
    render();
    nodes.forEach((n) => positionCache.set(n.id, { x: n.x!, y: n.y! }));
    if (reuse && keepView) svg.call(zoom.transform as never, keepView);
    else fit(false);
    return () => {
      sim.stop();
      nodes.forEach((n) => n.x != null && positionCache.set(n.id, { x: n.x, y: n.y! }));
    };
  }, [filtered, width, height]);

  // Selection / emphasis styling without rebuilding the layout.
  useEffect(() => {
    const svg = d3.select(svgRef.current!);
    svg.selectAll<SVGGElement, SimNode>('g.node')
      .attr('opacity', (d) => (!emphasis || emphasis.has(d.id) || d.kind === 'identity' ? 1 : 0.18))
      .select('circle.body')
      .attr('stroke', (d) => (d.id === selectedId ? 'var(--ink)' : '#fff'))
      .attr('stroke-width', (d) => (d.id === selectedId ? 3 : 1.8));
    svg.selectAll<SVGLineElement, SimLink>('line.edge').attr('stroke-opacity', (d) => {
      if (!emphasis) return 0.85;
      const s = typeof d.source === 'string' ? d.source : d.source.id;
      const t = typeof d.target === 'string' ? d.target : d.target.id;
      return emphasis.has(s) && emphasis.has(t) ? 1 : 0.08;
    });
  }, [selectedId, emphasis, filtered]);

  const zoomBy = (k: number) => {
    const svg = d3.select(svgRef.current!);
    if (zoomRef.current) svg.transition().duration(200).call(zoomRef.current.scaleBy as never, k);
  };

  return (
    <div className="graph-wrap" ref={wrapRef} style={{ height }}>
      <svg ref={svgRef} width={width} height={height} role="img" aria-label="Payment network graph" />
      {filtered.nodes.length === 0 && <div className="graph-empty">No accounts to show for this view yet.</div>}
      <div className="graph-tools">
        <button className="icon-btn" onClick={() => zoomBy(1.3)} aria-label="Zoom in"><Plus size={15} /></button>
        <button className="icon-btn" onClick={() => zoomBy(1 / 1.3)} aria-label="Zoom out"><Minus size={15} /></button>
        <button className="icon-btn" onClick={() => (svgRef.current as unknown as { __fit?: () => void }).__fit?.()} aria-label="Fit to view"><Maximize2 size={14} /></button>
      </div>
      <div className="graph-legend" aria-hidden>
        <span><i className="lg-dot" style={{ background: 'var(--bad)' }} /> High ≥70%</span>
        <span><i className="lg-dot" style={{ background: '#d59a1b' }} /> Elevated ≥30%</span>
        <span><i className="lg-dot" style={{ background: '#8c8f96' }} /> Low</span>
        <span><i className="lg-dot" style={{ background: '#8f1d1d', outline: '1.5px dashed #8f1d1d', outlineOffset: 1 }} /> Confirmed</span>
        {showIdentity && <span><i className="lg-dot" style={{ background: 'var(--identity)', borderRadius: 2, transform: 'rotate(45deg)' }} /> Shared identity</span>}
        <span><i className="lg-dot" style={{ background: 'var(--info)', width: 6, height: 6 }} /> In pattern</span>
        <span style={{ color: 'var(--accent)' }}>━ Flagged payment</span>
        <span className="faint">Ctrl/⌘ + scroll to zoom</span>
      </div>
      {tip && (
        <div className="graph-tip" style={{ left: Math.min(tip.x + 14, width - 270), top: Math.max(8, tip.y - 10) }}>{tip.html}</div>
      )}
    </div>
  );
}
