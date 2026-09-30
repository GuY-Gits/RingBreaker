import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { motion } from 'framer-motion';
import { RefreshCw, ZoomIn, ZoomOut, Maximize2, Minimize2, X, RotateCcw } from 'lucide-react';

// ============================================
// TYPES — RingBreaker graph snapshot (F14)
// ============================================

export interface SnapshotNode {
  id: string;
  risk: number;
  role?: string;
  label?: string;
}

export interface SnapshotEdge {
  source: string;
  target: string;
  amount?: number;
  weight?: number;
  timestamp?: string;
}

export interface GraphSnapshot {
  nodes: SnapshotNode[];
  edges: SnapshotEdge[];
}

export interface GraphLink {
  source: string;
  target: string;
  amount: number;
  timestamp?: string;
}

export interface GraphNode {
  id: string;
  label: string;
  risk: number;
  role?: string;
  totalSent?: number;
  totalReceived?: number;
  transactionCount?: number;
}

export interface TransactionGraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

interface NetworkGraphProps {
  snapshot: GraphSnapshot;
  highlightNodeIds?: string[];
  highlightEdgeKeys?: string[];
  isLoading?: boolean;
  onRefresh?: () => void;
  /** When omitted, follows `prefers-color-scheme`. */
  isDark?: boolean;
}

interface SimulationNode extends d3.SimulationNodeDatum, GraphNode {
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
  radius: number;
  highlighted?: boolean;
}

interface SimulationLink extends d3.SimulationLinkDatum<SimulationNode> {
  amount: number;
  timestamp?: string;
  index?: number;
  highlighted?: boolean;
  curveOffset?: number;
}

// ============================================
// TRANSFORM SNAPSHOT → GRAPH DATA
// ============================================

const transformSnapshotToGraph = (snapshot: GraphSnapshot): TransactionGraphData => {
  if (!snapshot?.nodes?.length) {
    return { nodes: [], links: [] };
  }

  const nodes: GraphNode[] = snapshot.nodes.map((n) => ({
    id: n.id,
    label: n.label ?? n.id,
    risk: n.risk ?? 0,
    role: n.role,
    totalSent: 0,
    totalReceived: 0,
    transactionCount: 0,
  }));

  const nodeMap = new Map(nodes.map((n) => [n.id, n]));
  const links: GraphLink[] = [];

  for (const e of snapshot.edges ?? []) {
    const amount = e.amount ?? e.weight ?? 1;
    const src = nodeMap.get(e.source);
    const tgt = nodeMap.get(e.target);
    if (src && tgt) {
      src.totalSent = (src.totalSent ?? 0) + amount;
      tgt.totalReceived = (tgt.totalReceived ?? 0) + amount;
      src.transactionCount = (src.transactionCount ?? 0) + 1;
      tgt.transactionCount = (tgt.transactionCount ?? 0) + 1;
    }
    links.push({
      source: e.source,
      target: e.target,
      amount,
      timestamp: e.timestamp,
    });
  }

  return { nodes, links };
};

// ============================================
// RISK COLOUR (F14 — colour nodes by risk)
// ============================================

const riskColor = (risk: number): string => {
  const t = Math.max(0, Math.min(1, risk));
  const low = d3.rgb(34, 197, 94);
  const mid = d3.rgb(234, 179, 8);
  const high = d3.rgb(239, 68, 68);
  if (t < 0.5) {
    return d3.interpolateRgb(low, mid)(t * 2).formatHex();
  }
  return d3.interpolateRgb(mid, high)((t - 0.5) * 2).formatHex();
};

const edgeKey = (source: string, target: string) => `${source}->${target}`;

// ============================================
// NETWORK GRAPH COMPONENT
// ============================================

export const NetworkGraph = ({
  snapshot,
  highlightNodeIds = [],
  highlightEdgeKeys = [],
  isLoading = false,
  onRefresh,
  isDark: isDarkProp,
}: NetworkGraphProps) => {
  const highlightNodes = useMemo(() => new Set(highlightNodeIds), [highlightNodeIds]);
  const highlightEdges = useMemo(() => new Set(highlightEdgeKeys), [highlightEdgeKeys]);

  const [systemDark, setSystemDark] = useState(
    () => typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: dark)').matches
  );

  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = () => setSystemDark(mq.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  const isDark = isDarkProp ?? systemDark;

  const data = useMemo(() => transformSnapshotToGraph(snapshot ?? { nodes: [], edges: [] }), [snapshot]);

  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const [dimensions, setDimensions] = useState({ width: 600, height: 500 });
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const { width, height } = containerRef.current.getBoundingClientRect();
        setDimensions({ width: width || 600, height: height || 500 });
      }
    };

    const timeoutId = setTimeout(updateDimensions, 100);
    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => {
      window.removeEventListener('resize', updateDimensions);
      clearTimeout(timeoutId);
    };
  }, [isFullscreen]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isFullscreen) {
        setIsFullscreen(false);
      }
    };

    if (isFullscreen) {
      document.addEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = '';
    };
  }, [isFullscreen]);

  useEffect(() => {
    if (!svgRef.current || !data.nodes.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const { width, height } = dimensions;
    const centerX = width / 2;
    const centerY = height / 2;

    const colors = {
      nodeStroke: isDark ? '#475569' : '#94a3b8',
      nodeLabel: isDark ? '#e2e8f0' : '#1e293b',
      nodeSecondaryLabel: isDark ? '#64748b' : '#94a3b8',
      tooltipBg: isDark ? '#1e293b' : '#ffffff',
      tooltipBorder: isDark ? '#475569' : '#e2e8f0',
      tooltipText: isDark ? '#94a3b8' : '#64748b',
      dimOpacity: 0.25,
    };

    const nodes: SimulationNode[] = data.nodes.map((n) => {
      const highlighted = highlightNodes.has(n.id);
      return {
        ...n,
        highlighted,
        radius: highlighted ? 26 : 20,
      };
    });

    const links: SimulationLink[] = data.links.map((l, i) => {
      const highlighted =
        highlightEdges.has(edgeKey(l.source, l.target)) ||
        (highlightNodes.has(l.source) && highlightNodes.has(l.target));
      return {
        source: l.source,
        target: l.target,
        amount: l.amount,
        timestamp: l.timestamp,
        index: i,
        highlighted,
      };
    });

    const linkGroups = new Map<string, SimulationLink[]>();
    links.forEach((link) => {
      const s = typeof link.source === 'string' ? link.source : (link.source as SimulationNode).id;
      const t = typeof link.target === 'string' ? link.target : (link.target as SimulationNode).id;
      const key = [s, t].sort().join('-');
      if (!linkGroups.has(key)) {
        linkGroups.set(key, []);
      }
      linkGroups.get(key)!.push(link);
    });

    linkGroups.forEach((group) => {
      const count = group.length;
      group.forEach((link, i) => {
        link.curveOffset = count === 1 ? 0 : (i - (count - 1) / 2) * 25;
      });
    });

    const container = svg.append('g').attr('class', 'graph-container');

    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.2, 4])
      .on('zoom', (event) => {
        container.attr('transform', event.transform);
      });

    zoomRef.current = zoom;

    svg.call(zoom).on('dblclick.zoom', null);

    const initialTransform = d3.zoomIdentity
      .translate(centerX, centerY)
      .scale(0.9)
      .translate(-centerX, -centerY);
    svg.call(zoom.transform, initialTransform);

    const defs = svg.append('defs');

    const alertGlow = defs
      .append('filter')
      .attr('id', 'alertGlow')
      .attr('x', '-50%')
      .attr('y', '-50%')
      .attr('width', '200%')
      .attr('height', '200%');

    alertGlow.append('feGaussianBlur').attr('stdDeviation', '4').attr('result', 'coloredBlur');
    const alertMerge = alertGlow.append('feMerge');
    alertMerge.append('feMergeNode').attr('in', 'coloredBlur');
    alertMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    defs
      .append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 20)
      .attr('refY', 0)
      .attr('markerWidth', 5)
      .attr('markerHeight', 5)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', isDark ? '#94a3b8' : '#64748b');

    const simulation = d3
      .forceSimulation<SimulationNode>(nodes)
      .force(
        'link',
        d3
          .forceLink<SimulationNode, SimulationLink>(links)
          .id((d) => d.id)
          .distance(150)
          .strength(0.3)
      )
      .force('charge', d3.forceManyBody().strength(-500))
      .force('center', d3.forceCenter(centerX, centerY))
      .force('collision', d3.forceCollide().radius(60))
      .force('x', d3.forceX(centerX).strength(0.05))
      .force('y', d3.forceY(centerY).strength(0.05));

    const hasHighlight = highlightNodes.size > 0 || highlightEdges.size > 0;

    const link = container
      .append('g')
      .attr('class', 'links')
      .selectAll('path')
      .data(links)
      .join('path')
      .attr('fill', 'none')
      .attr('stroke', (d) => (d.highlighted ? '#f97316' : isDark ? '#64748b' : '#94a3b8'))
      .attr('stroke-opacity', (d) => {
        if (!hasHighlight) return 0.7;
        return d.highlighted ? 1 : colors.dimOpacity;
      })
      .attr('stroke-width', (d) =>
        d.highlighted ? 4 : Math.min(Math.max(d.amount / 500, 1.5), 4)
      )
      .attr('marker-end', 'url(#arrowhead)')
      .style('cursor', 'pointer')
      .on('mouseover', function (_event, d) {
        d3.select(this).attr('stroke-opacity', 1).attr('stroke-width', d.highlighted ? 5 : 4);
      })
      .on('mouseout', function (_event, d) {
        d3.select(this)
          .attr('stroke-opacity', !hasHighlight || d.highlighted ? (d.highlighted ? 1 : 0.7) : colors.dimOpacity)
          .attr('stroke-width', d.highlighted ? 4 : Math.min(Math.max(d.amount / 500, 1.5), 4));
      });

    const node = container
      .append('g')
      .attr('class', 'nodes')
      .selectAll<SVGGElement, SimulationNode>('g')
      .data(nodes)
      .join('g');

    node.call(
      d3
        .drag<SVGGElement, SimulationNode>()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended)
    );

    node
      .append('circle')
      .attr('r', (d) => d.radius)
      .attr('fill', (d) => riskColor(d.risk))
      .attr('stroke', (d) => (d.highlighted ? '#f97316' : colors.nodeStroke))
      .attr('stroke-width', (d) => (d.highlighted ? 3 : 2))
      .attr('filter', (d) => (d.highlighted ? 'url(#alertGlow)' : null))
      .attr('opacity', (d) => {
        if (!hasHighlight) return 1;
        return d.highlighted ? 1 : colors.dimOpacity;
      })
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        event.stopPropagation();
        setSelectedNode(d);
      })
      .on('mouseover', function (_event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', d.radius + 5)
          .attr('stroke', d.highlighted ? '#fb923c' : '#8b5cf6');
      })
      .on('mouseout', function (_event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', d.radius)
          .attr('stroke', d.highlighted ? '#f97316' : colors.nodeStroke);
      });

    node
      .append('text')
      .attr('class', 'node-label')
      .attr('dy', (d) => d.radius + 15)
      .attr('text-anchor', 'middle')
      .attr('fill', colors.nodeLabel)
      .attr('font-size', '11px')
      .attr('font-weight', (d) => (d.highlighted ? '600' : '400'))
      .style('pointer-events', 'none')
      .text((d) => {
        const name = d.label || d.id;
        return name.length > 12 ? name.substring(0, 12) + '...' : name;
      });

    node
      .append('text')
      .attr('class', 'node-risk')
      .attr('dy', (d) => -(d.radius + 8))
      .attr('text-anchor', 'middle')
      .attr('fill', colors.nodeSecondaryLabel)
      .attr('font-size', '9px')
      .style('pointer-events', 'none')
      .text((d) => `risk ${(d.risk * 100).toFixed(0)}%`);

    simulation.on('tick', () => {
      link.attr('d', (d) => {
        const source = d.source as SimulationNode;
        const target = d.target as SimulationNode;
        const dx = target.x! - source.x!;
        const dy = target.y! - source.y!;
        const curveOffset = d.curveOffset || 0;

        if (curveOffset === 0) {
          return `M${source.x},${source.y}L${target.x},${target.y}`;
        }

        const midX = (source.x! + target.x!) / 2;
        const midY = (source.y! + target.y!) / 2;
        const angle = Math.atan2(dy, dx);
        const offsetX = midX + curveOffset * Math.cos(angle + Math.PI / 2);
        const offsetY = midY + curveOffset * Math.sin(angle + Math.PI / 2);

        return `M${source.x},${source.y}Q${offsetX},${offsetY} ${target.x},${target.y}`;
      });

      node.attr('transform', (d) => `translate(${d.x},${d.y})`);
    });

    function dragstarted(
      event: d3.D3DragEvent<SVGGElement, SimulationNode, SimulationNode>,
      d: SimulationNode
    ) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(
      event: d3.D3DragEvent<SVGGElement, SimulationNode, SimulationNode>,
      d: SimulationNode
    ) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(
      event: d3.D3DragEvent<SVGGElement, SimulationNode, SimulationNode>,
      d: SimulationNode
    ) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }

    svg.on('click', () => setSelectedNode(null));

    return () => {
      simulation.stop();
    };
  }, [data, dimensions, isDark, highlightNodes, highlightEdges]);

  const formatAmount = (amount: number): string => {
    if (amount >= 100000) return (amount / 100000).toFixed(1) + 'L';
    if (amount >= 1000) return (amount / 1000).toFixed(1) + 'K';
    return amount.toString();
  };

  const handleZoom = (factor: number) => {
    if (!svgRef.current || !zoomRef.current) return;
    const svg = d3.select(svgRef.current);
    const currentTransform = d3.zoomTransform(svgRef.current);
    const newScale = currentTransform.k * factor;

    svg.transition().duration(300).call(zoomRef.current.scaleTo, newScale);
  };

  const handleReset = () => {
    if (!svgRef.current || !zoomRef.current) return;
    const svg = d3.select(svgRef.current);
    const { width, height } = dimensions;
    const centerX = width / 2;
    const centerY = height / 2;

    svg
      .transition()
      .duration(500)
      .call(
        zoomRef.current.transform,
        d3.zoomIdentity.translate(centerX, centerY).scale(0.9).translate(-centerX, -centerY)
      );
  };

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-125 bg-slate-100 dark:bg-slate-900/50 rounded-2xl border border-slate-200 dark:border-slate-800/50">
        <RefreshCw className="w-8 h-8 text-primary-500 animate-spin mb-4" />
        <p className="text-slate-600 dark:text-slate-400">Loading payment network...</p>
      </div>
    );
  }

  if (!data.nodes.length) {
    return (
      <div className="flex flex-col items-center justify-center h-125 bg-slate-100 dark:bg-slate-900/50 rounded-2xl border border-slate-200 dark:border-slate-800/50">
        <div className="w-16 h-16 rounded-full bg-slate-200 dark:bg-slate-800 flex items-center justify-center mb-4">
          <Maximize2 className="w-8 h-8 text-slate-400 dark:text-slate-600" />
        </div>
        <p className="text-slate-600 dark:text-slate-400 text-center">No network data yet</p>
        <p className="text-slate-500 text-sm text-center mt-2">Payments from the stream will appear here</p>
      </div>
    );
  }

  return (
    <div className={isFullscreen ? 'fixed inset-0 z-50 bg-white dark:bg-slate-950' : 'relative'}>
      {isFullscreen && (
        <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between p-4 bg-linear-to-b from-white dark:from-slate-950 to-transparent">
          <h2 className="text-lg font-semibold text-slate-900 dark:text-white">Live payment graph</h2>
          <button
            type="button"
            onClick={() => setIsFullscreen(false)}
            className="w-10 h-10 rounded-lg bg-slate-200/80 dark:bg-slate-800/80 backdrop-blur-sm border border-slate-300/50 dark:border-slate-700/50 flex items-center justify-center text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-300/80 dark:hover:bg-slate-700/80 transition-colors"
            title="Exit Fullscreen"
          >
            <X size={20} />
          </button>
        </div>
      )}

      <div
        ref={containerRef}
        className={`bg-slate-100 dark:bg-slate-900/50 rounded-2xl border border-slate-200 dark:border-slate-800/50 overflow-hidden ${
          isFullscreen ? 'h-full rounded-none border-0' : 'h-125'
        }`}
      >
        <svg
          ref={svgRef}
          width="100%"
          height="100%"
          className="bg-linear-to-br from-slate-50 via-slate-100 to-slate-50 dark:from-slate-950 dark:via-slate-900 dark:to-slate-950 cursor-grab active:cursor-grabbing touch-none"
        />

        <div className={`absolute ${isFullscreen ? 'top-16' : 'top-4'} right-4 flex flex-col gap-2`}>
          <button
            type="button"
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="w-10 h-10 rounded-lg bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border border-slate-200/50 dark:border-slate-700/50 flex items-center justify-center text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100/80 dark:hover:bg-slate-700/80 transition-colors"
            title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
          >
            {isFullscreen ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
          </button>
          <button
            type="button"
            onClick={() => handleZoom(1.2)}
            className="w-10 h-10 rounded-lg bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border border-slate-200/50 dark:border-slate-700/50 flex items-center justify-center text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100/80 dark:hover:bg-slate-700/80 transition-colors"
            title="Zoom In"
          >
            <ZoomIn size={18} />
          </button>
          <button
            type="button"
            onClick={() => handleZoom(0.8)}
            className="w-10 h-10 rounded-lg bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border border-slate-200/50 dark:border-slate-700/50 flex items-center justify-center text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100/80 dark:hover:bg-slate-700/80 transition-colors"
            title="Zoom Out"
          >
            <ZoomOut size={18} />
          </button>
          <button
            type="button"
            onClick={handleReset}
            className="w-10 h-10 rounded-lg bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border border-slate-200/50 dark:border-slate-700/50 flex items-center justify-center text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100/80 dark:hover:bg-slate-700/80 transition-colors"
            title="Reset View"
          >
            <RotateCcw size={18} />
          </button>
          {onRefresh && (
            <button
              type="button"
              onClick={onRefresh}
              className="w-10 h-10 rounded-lg bg-primary-500/20 backdrop-blur-sm border border-primary-500/30 flex items-center justify-center text-primary-400 hover:text-white hover:bg-primary-500/30 transition-colors"
              title="Refresh"
            >
              <RefreshCw size={18} />
            </button>
          )}
        </div>

        <div className="absolute bottom-4 left-4 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm rounded-lg p-3 border border-slate-200/50 dark:border-slate-700/50">
          <p className="text-xs text-slate-600 dark:text-slate-400 font-medium mb-2">Legend</p>
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-green-500" />
              <span className="text-xs text-slate-700 dark:text-slate-300">Low risk</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-amber-500" />
              <span className="text-xs text-slate-700 dark:text-slate-300">Medium risk</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500" />
              <span className="text-xs text-slate-700 dark:text-slate-300">High risk</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full ring-2 ring-orange-500 bg-orange-500/30" />
              <span className="text-xs text-slate-700 dark:text-slate-300">Alert subgraph</span>
            </div>
          </div>
          <p className="text-xs text-slate-500 mt-2">{data.links.length} payment edges</p>
        </div>
      </div>

      {selectedNode && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 20 }}
          className="absolute top-4 left-4 bg-white/95 dark:bg-slate-800/95 backdrop-blur-sm rounded-xl p-4 border border-slate-200/50 dark:border-slate-700/50 max-w-xs"
        >
          <div className="flex items-start gap-3">
            <div
              className="w-10 h-10 rounded-full flex items-center justify-center"
              style={{ backgroundColor: riskColor(selectedNode.risk) }}
            >
              <span className="text-white font-semibold">{selectedNode.label[0]?.toUpperCase()}</span>
            </div>
            <div className="flex-1 min-w-0">
              <h4 className="text-slate-900 dark:text-white font-medium truncate">{selectedNode.label}</h4>
              <p className="text-slate-600 dark:text-slate-400 text-sm truncate">{selectedNode.id}</p>
              {selectedNode.role && (
                <p className="text-slate-500 text-xs mt-1">Role: {selectedNode.role}</p>
              )}
              <p className="text-slate-500 text-xs">Risk: {(selectedNode.risk * 100).toFixed(1)}%</p>
            </div>
          </div>
          {(selectedNode.totalSent || selectedNode.totalReceived) && (
            <div className="mt-3 pt-3 border-t border-slate-200/50 dark:border-slate-700/50 grid grid-cols-2 gap-2">
              {selectedNode.totalSent !== undefined && (
                <div>
                  <p className="text-xs text-slate-500">Total sent</p>
                  <p className="text-sm font-medium text-red-400">₹{formatAmount(selectedNode.totalSent)}</p>
                </div>
              )}
              {selectedNode.totalReceived !== undefined && (
                <div>
                  <p className="text-xs text-slate-500">Total received</p>
                  <p className="text-sm font-medium text-green-400">₹{formatAmount(selectedNode.totalReceived)}</p>
                </div>
              )}
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
};

export default NetworkGraph;
