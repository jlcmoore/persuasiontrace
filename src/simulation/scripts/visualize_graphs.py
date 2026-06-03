"""
Generate an interactive HTML visualization for the generated belief graphs.
"""

import argparse
import json
import os
from collections import defaultdict

from simulation.io import read_jsonl_graphs

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <title>Belief Graph Visualization</title>
  <script type="text/javascript" src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
  <script type="text/javascript" src="https://unpkg.com/jspdf@2.5.1/dist/jspdf.umd.min.js"></script>
  <script type="text/javascript" src="https://unpkg.com/svg2pdf.js@2.5.0/dist/svg2pdf.umd.min.js"></script>
  <style type="text/css">
    body {
      font-family: sans-serif;
      margin: 20px;
    }
    #mynetwork {
      position: relative;
      width: 100%;
      height: 800px;
      border: 1px solid lightgray;
      background-color: #fafafa;
      overflow: hidden;
    }
    .controls {
      margin-bottom: 15px;
      padding: 10px;
      background-color: #eee;
      border-radius: 5px;
    }
    select {
      width: 80%;
      padding: 5px;
      font-size: 16px;
    }
    .control-row {
      margin-top: 10px;
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }
    .small-select {
      width: 160px;
    }
    .wide-select {
      width: 210px;
    }
    .small-input {
      width: 80px;
      padding: 4px;
      font-size: 14px;
    }
    #edit-status {
      font-size: 13px;
      color: #1f2937;
      min-height: 18px;
    }
    button {
      padding: 6px 12px;
      cursor: pointer;
    }
    .graph-svg {
      width: 100%;
      height: 100%;
      display: block;
    }
    .node-rect {
      rx: 6;
      ry: 6;
    }
    #node-tooltip {
      position: absolute;
      z-index: 20;
      display: none;
      max-width: 420px;
      background: #ffffff;
      border: 1px solid #cccccc;
      border-radius: 6px;
      box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
      padding: 10px;
      font-size: 12px;
      pointer-events: none;
    }
    #edit-action-menu {
      position: absolute;
      z-index: 30;
      display: none;
      background: #ffffff;
      border: 1px solid #cfcfcf;
      border-radius: 6px;
      box-shadow: 0 6px 14px rgba(0, 0, 0, 0.16);
      padding: 6px;
      min-width: 180px;
    }
    #edit-action-menu button {
      width: 100%;
      text-align: left;
      margin: 2px 0;
      padding: 6px 8px;
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 4px;
      font-size: 13px;
      cursor: pointer;
    }
    #edit-action-menu button:hover {
      background: #f3f4f6;
    }
  </style>
</head>
<body>
  <h2>Belief Graph Viewer</h2>
  <div class="controls">
    <label for="graph-select"><strong>Select Graph:</strong></label><br/>
    <select id="graph-select" onchange="switchGraph(this.value)"></select>
    <div class="control-row">
      <label>
        <input type="checkbox" id="hide-prior-hint" onchange="redrawCurrentGraph()"/>
        Hide hover hints
      </label>
      <label for="font-size">Font size</label>
      <input
        id="font-size"
        class="small-input"
        type="number"
        min="8"
        max="40"
        step="1"
        value="16"
        onchange="redrawCurrentGraph()"
      />
      <select id="export-format" class="small-select">
        <option value="pdf">PDF</option>
        <option value="svg">SVG</option>
      </select>
      <button type="button" onclick="exportCurrentView()">Export Current View</button>
    </div>
    <div class="control-row">
      <label for="edit-mode"><strong>Edit mode</strong></label>
      <select id="edit-mode" class="wide-select" onchange="onEditModeChange()">
        <option value="select">Select</option>
        <option value="add-edge">Add edge</option>
      </select>
      <label for="edge-sign">New edge sign</label>
      <select id="edge-sign" class="small-select">
        <option value="positive">Positive (+)</option>
        <option value="negative">Negative (-)</option>
      </select>
      <button type="button" onclick="undoEdit()">Undo</button>
      <button type="button" onclick="redoEdit()">Redo</button>
      <button type="button" onclick="exportAllGraphsJsonl()">Export All Edited JSONL</button>
    </div>
    <div id="edit-status"></div>
  </div>
  <div id="mynetwork">
    <div id="node-tooltip"></div>
    <div id="edit-action-menu"></div>
  </div>

  <script type="text/javascript">
    const graphsData = __GRAPH_DATA__;
    let currentGraphIndex = 0;
    let pendingEdgeSource = null;
    let undoStack = [];
    let redoStack = [];

    function populateSelector() {
      const select = document.getElementById("graph-select");
      graphsData.forEach((graph, index) => {
        const option = document.createElement("option");
        option.value = index;
        const label = graph.target.length > 100
          ? graph.target.substring(0, 100) + "..."
          : graph.target;
        option.text = `Graph ${index + 1}: ${label}`;
        select.appendChild(option);
      });
    }

    function stripHtmlTags(rawText) {
      return String(rawText || "").replace(/<[^>]*>/g, "");
    }

    function escapeXml(text) {
      return String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&apos;");
    }

    function splitLongWord(word, measure, maxWidth) {
      const parts = [];
      let current = "";
      for (const ch of word) {
        const next = current + ch;
        if (current && measure(next) > maxWidth) {
          parts.push(current);
          current = ch;
        } else {
          current = next;
        }
      }
      if (current) {
        parts.push(current);
      }
      return parts;
    }

    function wrapTextLines(rawText, fontSize, maxInnerWidthPx) {
      const canvas = document.createElement("canvas");
      const context = canvas.getContext("2d");
      context.font = `${fontSize}px sans-serif`;
      const measure = (text) => context.measureText(text).width;

      const lines = [];
      const paragraphs = String(rawText || "").split("\\n");
      paragraphs.forEach((paragraph) => {
        const words = paragraph.split(/\\s+/).filter(Boolean);
        if (words.length === 0) {
          lines.push("");
          return;
        }
        let current = "";
        words.forEach((word) => {
          if (measure(word) > maxInnerWidthPx) {
            const broken = splitLongWord(word, measure, maxInnerWidthPx);
            broken.forEach((chunk) => {
              if (current) {
                lines.push(current);
              }
              current = chunk;
            });
            return;
          }
          const next = current ? `${current} ${word}` : word;
          if (measure(next) <= maxInnerWidthPx) {
            current = next;
          } else {
            if (current) {
              lines.push(current);
            }
            current = word;
          }
        });
        if (current) {
          lines.push(current);
        }
      });
      return lines;
    }

    function clamp(value, minValue, maxValue) {
      return Math.max(minValue, Math.min(maxValue, value));
    }

    function mapNodeForDisplay(node, hideHoverHints, fontSize) {
      let label = String(node.label || "");
      if (hideHoverHints) {
        label = label
          .replace("\\n<i>(Hover for prior)</i>", "")
          .replace("\\n<i>(Hover for CPT)</i>", "");
      }
      const plainLabel = stripHtmlTags(label);
      const wrappedLines = wrapTextLines(plainLabel, fontSize, 280);
      const canvas = document.createElement("canvas");
      const context = canvas.getContext("2d");
      context.font = `${fontSize}px sans-serif`;
      const maxLineWidth = wrappedLines.reduce((acc, line) => {
        return Math.max(acc, context.measureText(line).width);
      }, 0);
      const width = clamp(Math.ceil(maxLineWidth + 24), 150, 300);
      const lineHeight = Math.max(12, Math.round(fontSize * 1.2));
      const height = Math.max(48, Math.ceil((wrappedLines.length * lineHeight) + 20));
      return {
        id: String(node.id),
        lines: wrappedLines,
        width: width,
        height: height,
        background: node.color?.background || "#ffffff",
        border: node.color?.border || "#999999",
        borderWidth: Number(node.borderWidth || 1),
        tooltipHtml: String(node.title || ""),
      };
    }

    function mapEdgeForDisplay(edge) {
      return {
        from: String(edge.from),
        to: String(edge.to),
        label: stripHtmlTags(edge.label || ""),
        color: edge.color?.color || "#999999",
        dashed: Boolean(edge.dashes),
        width: Number(edge.width || 2),
      };
    }

    function drawArrowHead(parentGroup, x1, y1, x2, y2, color) {
      const angle = Math.atan2(y2 - y1, x2 - x1);
      const size = 9;
      const leftX = x2 - (size * Math.cos(angle - Math.PI / 6));
      const leftY = y2 - (size * Math.sin(angle - Math.PI / 6));
      const rightX = x2 - (size * Math.cos(angle + Math.PI / 6));
      const rightY = y2 - (size * Math.sin(angle + Math.PI / 6));
      const triangle = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      triangle.setAttribute("points", `${x2},${y2} ${leftX},${leftY} ${rightX},${rightY}`);
      triangle.setAttribute("fill", color);
      parentGroup.appendChild(triangle);
    }

    function showTooltip(event, htmlContent) {
      const tooltip = document.getElementById("node-tooltip");
      if (!htmlContent) {
        return;
      }
      tooltip.innerHTML = htmlContent;
      tooltip.style.display = "block";
      moveTooltip(event);
    }

    function moveTooltip(event) {
      const container = document.getElementById("mynetwork");
      const tooltip = document.getElementById("node-tooltip");
      if (tooltip.style.display !== "block") {
        return;
      }
      const containerRect = container.getBoundingClientRect();
      const x = event.clientX - containerRect.left + 14;
      const y = event.clientY - containerRect.top + 14;
      tooltip.style.left = `${x}px`;
      tooltip.style.top = `${y}px`;
    }

    function hideTooltip() {
      const tooltip = document.getElementById("node-tooltip");
      tooltip.style.display = "none";
      tooltip.innerHTML = "";
    }

    function hideActionMenu() {
      const menu = document.getElementById("edit-action-menu");
      if (!menu) {
        return;
      }
      menu.style.display = "none";
      menu.innerHTML = "";
    }

    function showActionMenu(event, items) {
      const menu = document.getElementById("edit-action-menu");
      const container = document.getElementById("mynetwork");
      if (!menu || !container || !Array.isArray(items) || !items.length) {
        return;
      }
      hideActionMenu();
      event.stopPropagation();
      menu.innerHTML = "";
      items.forEach((item) => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = item.label;
        button.addEventListener("click", (clickEvent) => {
          clickEvent.stopPropagation();
          hideActionMenu();
          item.onClick();
        });
        menu.appendChild(button);
      });

      const containerRect = container.getBoundingClientRect();
      const x = event.clientX - containerRect.left + 8;
      const y = event.clientY - containerRect.top + 8;
      menu.style.display = "block";
      const maxX = Math.max(8, container.clientWidth - menu.offsetWidth - 8);
      const maxY = Math.max(8, container.clientHeight - menu.offsetHeight - 8);
      menu.style.left = `${Math.max(8, Math.min(maxX, x))}px`;
      menu.style.top = `${Math.max(8, Math.min(maxY, y))}px`;
    }

    function deepClone(value) {
      return JSON.parse(JSON.stringify(value));
    }

    function updateStatus(message) {
      const status = document.getElementById("edit-status");
      if (status) {
        status.textContent = String(message || "");
      }
    }

    function normalizeCurrentGraph() {
      const graph = graphsData[currentGraphIndex];
      if (!graph) {
        return null;
      }
      const editable = graph.editable || {};
      if (!editable.bayesian_network || typeof editable.bayesian_network !== "object") {
        editable.bayesian_network = {};
      }
      const bn = editable.bayesian_network;
      if (!Array.isArray(bn.belief_nodes)) {
        bn.belief_nodes = [];
      }
      if (!Array.isArray(bn.edges)) {
        bn.edges = [];
      }
      if (!Array.isArray(bn.joint_distribution)) {
        bn.joint_distribution = [];
      }
      if (!bn.target || typeof bn.target !== "string") {
        bn.target = graph.target || editable.id || "Target Proposition";
      }
      if (!bn.target_proposition || typeof bn.target_proposition !== "string") {
        bn.target_proposition = bn.target;
      }
      if (!bn.bayesian_network || typeof bn.bayesian_network !== "object") {
        bn.bayesian_network = {};
      }
      if (!Object.keys(bn.bayesian_network).length && bn.joint_distribution.length) {
        fitCptsFromJoint(bn);
      }
      graph.editable = editable;
      return graph;
    }

    function getCurrentStructure() {
      const graph = normalizeCurrentGraph();
      if (!graph || !graph.editable) {
        return null;
      }
      return graph.editable.bayesian_network;
    }

    function nodeNameForIndex(nodeIndex) {
      if (nodeIndex === 0) {
        return "Target";
      }
      return `Belief_${nodeIndex}`;
    }

    function addConditionKey(parentNames, values) {
      if (!parentNames.length) {
        return "prior";
      }
      return parentNames.map((name, idx) => `${name}=${values[idx]}`).join(",");
    }

    function productBooleans(count) {
      const rows = [];
      const total = Math.pow(2, count);
      for (let mask = 0; mask < total; mask += 1) {
        const row = [];
        for (let idx = 0; idx < count; idx += 1) {
          const bit = (mask >> (count - idx - 1)) & 1;
          row.push(Boolean(bit));
        }
        rows.push(row);
      }
      return rows;
    }

    function fitCptsFromJoint(structure) {
      if (!Array.isArray(structure.joint_distribution) || !structure.joint_distribution.length) {
        structure.bayesian_network = {};
        return false;
      }
      const numBeliefs = Array.isArray(structure.belief_nodes)
        ? structure.belief_nodes.length
        : 0;
      const allNodes = ["Target"];
      for (let idx = 1; idx <= numBeliefs; idx += 1) {
        allNodes.push(`Belief_${idx}`);
      }
      const parentsMap = {};
      allNodes.forEach((node) => {
        parentsMap[node] = [];
      });
      (structure.edges || []).forEach((edge) => {
        const sourceName = nodeNameForIndex(Number(edge.from));
        const targetName = nodeNameForIndex(Number(edge.to));
        if (!parentsMap[targetName]) {
          parentsMap[targetName] = [];
        }
        parentsMap[targetName].push(sourceName);
      });

      const cpts = {};
      allNodes.forEach((nodeName) => {
        const parents = (parentsMap[nodeName] || []).slice().sort();
        const marginals = {};
        const jointTrue = {};
        structure.joint_distribution.forEach((entry) => {
          const state = entry.state || {};
          const prob = Number(entry.probability || 0);
          const vals = parents.map((parent) => Boolean(state[parent]));
          const key = JSON.stringify(vals);
          marginals[key] = (marginals[key] || 0) + prob;
          if (Boolean(state[nodeName])) {
            jointTrue[key] = (jointTrue[key] || 0) + prob;
          }
        });

        const probs = {};
        productBooleans(parents.length).forEach((vals) => {
          const key = JSON.stringify(vals);
          const denom = Number(marginals[key] || 0);
          const numer = Number(jointTrue[key] || 0);
          const pTrue = denom > 0 ? numer / denom : 0.5;
          probs[addConditionKey(parents, vals)] = pTrue;
        });
        cpts[nodeName] = { parents: parents, probabilities: probs };
      });

      structure.bayesian_network = cpts;
      return true;
    }

    function marginalizeJointDistributionAfterNodeDelete(structure, deletedNodeIndex) {
      if (!Array.isArray(structure.joint_distribution) || !structure.joint_distribution.length) {
        structure.joint_distribution = [];
        return;
      }
      const oldCount = structure.belief_nodes.length + 1;
      const accum = {};
      structure.joint_distribution.forEach((entry) => {
        const state = entry.state || {};
        const newState = { Target: Boolean(state.Target) };
        let newBeliefIndex = 1;
        for (let oldBeliefIndex = 1; oldBeliefIndex <= oldCount; oldBeliefIndex += 1) {
          if (oldBeliefIndex === deletedNodeIndex) {
            continue;
          }
          const oldName = `Belief_${oldBeliefIndex}`;
          const newName = `Belief_${newBeliefIndex}`;
          newState[newName] = Boolean(state[oldName]);
          newBeliefIndex += 1;
        }
        const key = JSON.stringify(newState);
        accum[key] = (accum[key] || 0) + Number(entry.probability || 0);
      });
      const rows = Object.entries(accum).map(([key, probability]) => {
        return { state: JSON.parse(key), probability: probability };
      });
      rows.sort((left, right) => Number(right.probability) - Number(left.probability));
      structure.joint_distribution = rows;
    }

    function edgeExists(edges, source, target) {
      return edges.some((edge) => Number(edge.from) === source && Number(edge.to) === target);
    }

    function hasDirectedCycle(numBeliefs, edges) {
      const nodeCount = numBeliefs + 1;
      const adjacency = {};
      for (let node = 0; node <= numBeliefs; node += 1) {
        adjacency[node] = [];
      }
      edges.forEach((edge) => {
        const from = Number(edge.from);
        const to = Number(edge.to);
        if (from >= 0 && from <= numBeliefs && to >= 0 && to <= numBeliefs) {
          adjacency[from].push(to);
        }
      });
      const state = {};
      const VISITING = 1;
      const DONE = 2;

      function dfs(node) {
        state[node] = VISITING;
        for (const next of adjacency[node]) {
          if (state[next] === VISITING) {
            return true;
          }
          if (state[next] !== DONE && dfs(next)) {
            return true;
          }
        }
        state[node] = DONE;
        return false;
      }

      for (let node = 0; node < nodeCount; node += 1) {
        if (!state[node] && dfs(node)) {
          return true;
        }
      }
      return false;
    }

    function pushHistory() {
      const structure = getCurrentStructure();
      if (!structure) {
        return;
      }
      undoStack.push(deepClone(structure));
      if (undoStack.length > 50) {
        undoStack.shift();
      }
      redoStack = [];
    }

    function applyStructureSnapshot(snapshot) {
      const graph = normalizeCurrentGraph();
      if (!graph || !graph.editable) {
        return;
      }
      graph.editable.bayesian_network = deepClone(snapshot);
      redrawCurrentGraph();
    }

    function undoEdit() {
      const structure = getCurrentStructure();
      if (!structure || !undoStack.length) {
        updateStatus("Nothing to undo.");
        return;
      }
      redoStack.push(deepClone(structure));
      const prior = undoStack.pop();
      applyStructureSnapshot(prior);
      updateStatus("Undid last edit.");
    }

    function redoEdit() {
      const structure = getCurrentStructure();
      if (!structure || !redoStack.length) {
        updateStatus("Nothing to redo.");
        return;
      }
      undoStack.push(deepClone(structure));
      const next = redoStack.pop();
      applyStructureSnapshot(next);
      updateStatus("Redid last edit.");
    }

    function onEditModeChange() {
      pendingEdgeSource = null;
      hideActionMenu();
      const mode = document.getElementById("edit-mode").value;
      if (mode === "add-edge") {
        updateStatus("Add edge mode: click source node, then target node.");
        return;
      }
      updateStatus("Select mode: click any edge or node for actions.");
    }

    function switchGraph(index) {
      currentGraphIndex = Number(index);
      pendingEdgeSource = null;
      undoStack = [];
      redoStack = [];
      hideActionMenu();
      redrawCurrentGraph();
      updateStatus(`Viewing graph ${currentGraphIndex + 1}.`);
    }

    function deleteNodeByIndex(nodeId) {
      if (nodeId === 0) {
        updateStatus("Cannot delete the Target node.");
        return;
      }
      const structure = getCurrentStructure();
      if (!structure) {
        return;
      }
      const beliefIdx = Number(nodeId);
      if (beliefIdx < 1 || beliefIdx > structure.belief_nodes.length) {
        updateStatus("Invalid node selection.");
        return;
      }

      pushHistory();
      const updatedBeliefs = [];
      for (let idx = 1; idx <= structure.belief_nodes.length; idx += 1) {
        if (idx === beliefIdx) {
          continue;
        }
        updatedBeliefs.push(structure.belief_nodes[idx - 1]);
      }
      structure.belief_nodes = updatedBeliefs;

      const filteredEdges = [];
      (structure.edges || []).forEach((edge) => {
        const from = Number(edge.from);
        const to = Number(edge.to);
        if (from === beliefIdx || to === beliefIdx) {
          return;
        }
        filteredEdges.push({
          ...edge,
          from: from > beliefIdx ? from - 1 : from,
          to: to > beliefIdx ? to - 1 : to,
        });
      });
      structure.edges = filteredEdges;

      marginalizeJointDistributionAfterNodeDelete(structure, beliefIdx);
      fitCptsFromJoint(structure);
      redrawCurrentGraph();
      updateStatus(`Deleted Belief_${beliefIdx} and refit CPTs.`);
    }

    function addEdge(sourceId, targetId) {
      const structure = getCurrentStructure();
      if (!structure) {
        return;
      }
      if (sourceId === targetId) {
        updateStatus("Self edges are not allowed.");
        return;
      }
      if (edgeExists(structure.edges || [], sourceId, targetId)) {
        updateStatus("That edge already exists.");
        return;
      }
      const edgeSign = document.getElementById("edge-sign").value;
      const newEdge = {
        from: sourceId,
        to: targetId,
        positive_influence: edgeSign === "positive",
      };
      const candidateEdges = (structure.edges || []).concat([newEdge]);
      if (hasDirectedCycle(structure.belief_nodes.length, candidateEdges)) {
        updateStatus("Rejected: this edge would create a directed cycle.");
        return;
      }

      pushHistory();
      structure.edges = candidateEdges;
      fitCptsFromJoint(structure);
      redrawCurrentGraph();
      updateStatus(`Added edge ${nodeNameForIndex(sourceId)} -> ${nodeNameForIndex(targetId)}.`);
    }

    function deleteEdge(sourceId, targetId) {
      const structure = getCurrentStructure();
      if (!structure) {
        return;
      }
      const edges = structure.edges || [];
      const idx = edges.findIndex((edge) => {
        return Number(edge.from) === sourceId && Number(edge.to) === targetId;
      });
      if (idx < 0) {
        updateStatus("Edge not found.");
        return;
      }
      pushHistory();
      edges.splice(idx, 1);
      structure.edges = edges;
      fitCptsFromJoint(structure);
      redrawCurrentGraph();
      updateStatus(`Deleted edge ${nodeNameForIndex(sourceId)} -> ${nodeNameForIndex(targetId)}.`);
    }

    function flipEdgeSign(sourceId, targetId) {
      const structure = getCurrentStructure();
      if (!structure) {
        return;
      }
      const edges = structure.edges || [];
      const idx = edges.findIndex((edge) => {
        return Number(edge.from) === sourceId && Number(edge.to) === targetId;
      });
      if (idx < 0) {
        updateStatus("Edge not found.");
        return;
      }
      pushHistory();
      const current = edges[idx];
      current.positive_influence = !Boolean(current.positive_influence);
      fitCptsFromJoint(structure);
      redrawCurrentGraph();
      const signText = current.positive_influence ? "positive" : "negative";
      updateStatus(
        `Flipped sign for ${nodeNameForIndex(sourceId)} -> ${nodeNameForIndex(targetId)} (${signText}).`,
      );
    }

    function findEdge(sourceId, targetId) {
      const structure = getCurrentStructure();
      if (!structure) {
        return null;
      }
      const edge = (structure.edges || []).find((candidate) => {
        return Number(candidate.from) === sourceId && Number(candidate.to) === targetId;
      });
      return edge || null;
    }

    function handleNodeClick(nodeId, event) {
      const mode = document.getElementById("edit-mode").value;
      const parsedNodeId = Number(nodeId);
      if (!Number.isFinite(parsedNodeId)) {
        return;
      }
      if (mode === "add-edge") {
        hideActionMenu();
        if (pendingEdgeSource === null) {
          pendingEdgeSource = parsedNodeId;
          updateStatus(`Selected source: ${nodeNameForIndex(parsedNodeId)}. Click a target node.`);
          return;
        }
        const source = pendingEdgeSource;
        pendingEdgeSource = null;
        addEdge(source, parsedNodeId);
        return;
      }
      const items = [];
      if (parsedNodeId !== 0) {
        items.push({
          label: `Delete ${nodeNameForIndex(parsedNodeId)}`,
          onClick: () => deleteNodeByIndex(parsedNodeId),
        });
      }
      items.push({
        label: `Start new edge from ${nodeNameForIndex(parsedNodeId)}`,
        onClick: () => {
          document.getElementById("edit-mode").value = "add-edge";
          pendingEdgeSource = parsedNodeId;
          updateStatus(`Selected source: ${nodeNameForIndex(parsedNodeId)}. Click a target node.`);
        },
      });
      showActionMenu(event, items);
    }

    function handleEdgeClick(fromId, toId, event) {
      hideTooltip();
      const sourceId = Number(fromId);
      const targetId = Number(toId);
      const edge = findEdge(sourceId, targetId);
      if (!edge) {
        updateStatus("Edge not found.");
        return;
      }
      const signText = Boolean(edge.positive_influence) ? "positive" : "negative";
      showActionMenu(event, [
        {
          label: `Flip sign (currently ${signText})`,
          onClick: () => flipEdgeSign(sourceId, targetId),
        },
        {
          label: `Delete edge ${nodeNameForIndex(sourceId)} -> ${nodeNameForIndex(targetId)}`,
          onClick: () => deleteEdge(sourceId, targetId),
        },
      ]);
    }

    function formatCptTooltip(nodeName, bnData) {
      if (!bnData || !bnData[nodeName]) {
        return "No CPT available";
      }
      const cpt = bnData[nodeName];
      const parents = Array.isArray(cpt.parents) ? cpt.parents : [];
      const probs = cpt.probabilities || {};
      let html = "<div style='font-family: sans-serif; font-size: 12px;'>";
      html += `<strong>Node: ${escapeXml(nodeName)}</strong><br/><br/>`;
      html += "<table border='1' cellpadding='4' style='border-collapse: collapse;'>";
      if (!parents.length) {
        html += `<tr><th>Prior P(True)</th><td>${Number(probs.prior || 0).toFixed(3)}</td></tr>`;
      } else {
        html += "<tr>";
        parents.forEach((parentName) => {
          html += `<th style='background-color:#eee'>${escapeXml(parentName.replace("Belief_", "B"))}</th>`;
        });
        html += "<th style='background-color:#eee'>P(True)</th></tr>";
        Object.entries(probs).forEach(([cond, val]) => {
          html += "<tr>";
          cond.split(",").forEach((part) => {
            const boolVal = part.split("=")[1] || "False";
            const color = boolVal === "True" ? "green" : "red";
            html += `<td style='color:${color}; text-align:center;'>${escapeXml(boolVal[0])}</td>`;
          });
          html += `<td>${Number(val).toFixed(3)}</td></tr>`;
        });
      }
      html += "</table></div>";
      return html;
    }

    function calculateEdgeWeightLabel(sourceName, targetName, bnData) {
      if (!bnData || !bnData[targetName]) {
        return "";
      }
      const cpt = bnData[targetName];
      const parents = Array.isArray(cpt.parents) ? cpt.parents : [];
      const probs = cpt.probabilities || {};
      if (!parents.includes(sourceName)) {
        return "";
      }
      const paired = {};
      Object.entries(probs).forEach(([cond, val]) => {
        if (cond === "prior") {
          return;
        }
        let sourceVal = null;
        const otherConds = [];
        cond.split(",").forEach((part) => {
          const split = part.split("=");
          const parent = split[0];
          const boolVal = split[1];
          if (parent === sourceName) {
            sourceVal = boolVal;
          } else {
            otherConds.push(part);
          }
        });
        const otherKey = otherConds.join(",");
        if (!paired[otherKey]) {
          paired[otherKey] = {};
        }
        paired[otherKey][sourceVal] = Number(val);
      });
      const diffs = [];
      Object.values(paired).forEach((vals) => {
        if (Object.prototype.hasOwnProperty.call(vals, "True")
          && Object.prototype.hasOwnProperty.call(vals, "False")) {
          diffs.push(Number(vals.True) - Number(vals.False));
        }
      });
      if (!diffs.length) {
        return "";
      }
      const avgDiff = diffs.reduce((acc, item) => acc + item, 0) / diffs.length;
      const sign = avgDiff > 0 ? "+" : "";
      return `dP: ${sign}${avgDiff.toFixed(2)}`;
    }

    function buildDisplayData(graphData, hideHoverHints) {
      const structure = getCurrentStructure();
      if (!structure) {
        return { nodes: [], edges: [] };
      }
      const bnData = structure.bayesian_network || {};
      const targetText = structure.target || graphData.target || "Target Proposition";

      const nodes = [];
      let targetLabel = `<b>Target:</b>\\n${targetText}`;
      if (bnData && Object.keys(bnData).length) {
        targetLabel += "\\n<i>(Hover for CPT)</i>";
      }
      if (hideHoverHints) {
        targetLabel = targetLabel.replace("\\n<i>(Hover for CPT)</i>", "");
      }
      nodes.push({
        id: 0,
        label: targetLabel,
        color: { background: "#D2E5FF", border: "#2B7CE9" },
        borderWidth: 2,
        title: formatCptTooltip("Target", bnData),
      });

      (structure.belief_nodes || []).forEach((beliefText, idx) => {
        const beliefId = idx + 1;
        const nodeName = `Belief_${beliefId}`;
        let label = `<b>B${beliefId}:</b> ${beliefText}`;
        if (bnData && Object.keys(bnData).length) {
          label += "\\n<i>(Hover for prior)</i>";
        }
        if (hideHoverHints) {
          label = label.replace("\\n<i>(Hover for prior)</i>", "");
        }
        nodes.push({
          id: beliefId,
          label: label,
          color: { background: "#ffffff", border: "#999999" },
          borderWidth: 1,
          title: formatCptTooltip(nodeName, bnData),
        });
      });

      const edges = [];
      (structure.edges || []).forEach((edge) => {
        const from = Number(edge.from);
        const to = Number(edge.to);
        const isPositive = Boolean(edge.positive_influence);
        const sourceName = nodeNameForIndex(from);
        const targetName = nodeNameForIndex(to);
        const color = isPositive ? "#4CAF50" : "#F44336";
        const cptLabel = calculateEdgeWeightLabel(sourceName, targetName, bnData);
        edges.push({
          from: from,
          to: to,
          color: { color: color },
          dashes: !isPositive,
          width: 2,
          label: cptLabel || (isPositive ? "+" : "-"),
        });
      });

      return { nodes: nodes, edges: edges };
    }

    function drawGraph(index) {
      currentGraphIndex = Number(index);
      const graphData = normalizeCurrentGraph();
      if (!graphData) {
        return;
      }
      hideActionMenu();

      const hideHoverHints = document.getElementById("hide-prior-hint").checked;
      const fontSizeInput = document.getElementById("font-size");
      const parsedFontSize = Number(fontSizeInput.value);
      const fontSize = Number.isFinite(parsedFontSize)
        ? clamp(parsedFontSize, 8, 40)
        : 16;

      const visData = buildDisplayData(graphData, hideHoverHints);
      const nodes = visData.nodes.map((node) =>
        mapNodeForDisplay(node, hideHoverHints, fontSize)
      );
      const edges = visData.edges.map((edge) => mapEdgeForDisplay(edge));

      const layoutGraph = new dagre.graphlib.Graph();
      layoutGraph.setGraph({
        rankdir: "LR",
        nodesep: 150,
        ranksep: 260,
        marginx: 20,
        marginy: 20,
      });
      layoutGraph.setDefaultEdgeLabel(() => ({}));
      nodes.forEach((node) => {
        layoutGraph.setNode(node.id, { width: node.width, height: node.height });
      });
      edges.forEach((edge) => {
        layoutGraph.setEdge(edge.from, edge.to);
      });
      dagre.layout(layoutGraph);

      const container = document.getElementById("mynetwork");
      const existingSvg = container.querySelector("svg");
      if (existingSvg) {
        container.removeChild(existingSvg);
      }

      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("class", "graph-svg");
      svg.setAttribute("xmlns", "http://www.w3.org/2000/svg");
      svg.setAttribute("version", "1.1");
      svg.addEventListener("click", () => {
        hideActionMenu();
      });
      container.insertBefore(svg, document.getElementById("node-tooltip"));

      const graphGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
      graphGroup.setAttribute("id", "graph-content");
      svg.appendChild(graphGroup);

      const lineHeight = Math.max(12, Math.round(fontSize * 1.2));
      const nodesById = {};

      nodes.forEach((node) => {
        const pos = layoutGraph.node(node.id);
        nodesById[node.id] = {
          ...node,
          x: pos.x - (node.width / 2),
          y: pos.y - (node.height / 2),
          centerX: pos.x,
          centerY: pos.y,
        };
      });

      edges.forEach((edge) => {
        const edgeLayout = layoutGraph.edge(edge.from, edge.to);
        if (!edgeLayout) {
          return;
        }
        const pathPoints = edgeLayout.points || [];
        if (pathPoints.length < 2) {
          return;
        }
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        const pathD = pathPoints.map((pt, idx) => {
          const prefix = idx === 0 ? "M" : "L";
          return `${prefix} ${pt.x} ${pt.y}`;
        }).join(" ");
        path.setAttribute("d", pathD);
        path.setAttribute("fill", "none");
        path.setAttribute("stroke", edge.color);
        path.setAttribute("stroke-width", String(edge.width));
        if (edge.dashed) {
          path.setAttribute("stroke-dasharray", "6 4");
        }
        graphGroup.appendChild(path);

        const clickPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
        clickPath.setAttribute("d", pathD);
        clickPath.setAttribute("fill", "none");
        clickPath.setAttribute("stroke", "transparent");
        clickPath.setAttribute("stroke-width", "14");
        clickPath.setAttribute("style", "cursor:pointer;");
        clickPath.addEventListener("click", (event) => {
          event.stopPropagation();
          handleEdgeClick(edge.from, edge.to, event);
        });
        graphGroup.appendChild(clickPath);

        const last = pathPoints[pathPoints.length - 1];
        const prev = pathPoints[pathPoints.length - 2];
        drawArrowHead(graphGroup, prev.x, prev.y, last.x, last.y, edge.color);

        if (edge.label) {
          const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
          const totalLength = path.getTotalLength();
          const midPoint = path.getPointAtLength(totalLength / 2);
          label.setAttribute("x", String(midPoint.x));
          label.setAttribute("y", String(midPoint.y - 4));
          label.setAttribute("text-anchor", "middle");
          label.setAttribute("font-family", "sans-serif");
          label.setAttribute("font-size", String(fontSize));
          label.setAttribute("fill", "#000000");
          label.textContent = edge.label;
          graphGroup.appendChild(label);
        }
      });

      nodes.forEach((node) => {
        const placed = nodesById[node.id];
        const nodeGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
        nodeGroup.setAttribute("data-node-id", node.id);

        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("class", "node-rect");
        rect.setAttribute("x", String(placed.x));
        rect.setAttribute("y", String(placed.y));
        rect.setAttribute("width", String(placed.width));
        rect.setAttribute("height", String(placed.height));
        rect.setAttribute("fill", placed.background);
        rect.setAttribute("stroke", placed.border);
        rect.setAttribute("stroke-width", String(Math.max(1, placed.borderWidth)));
        nodeGroup.appendChild(rect);

        placed.lines.forEach((line, idx) => {
          const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
          text.setAttribute("x", String(placed.x + 10));
          text.setAttribute("y", String(placed.y + 12 + ((idx + 1) * lineHeight)));
          text.setAttribute("font-family", "sans-serif");
          text.setAttribute("font-size", String(fontSize));
          text.setAttribute("fill", "#000000");
          text.textContent = line;
          nodeGroup.appendChild(text);
        });

        if (placed.tooltipHtml) {
          nodeGroup.addEventListener("mouseenter", (event) => {
            showTooltip(event, placed.tooltipHtml);
          });
          nodeGroup.addEventListener("mousemove", (event) => {
            moveTooltip(event);
          });
          nodeGroup.addEventListener("mouseleave", () => {
            hideTooltip();
          });
        }
        nodeGroup.setAttribute("style", "cursor:pointer;");
        nodeGroup.addEventListener("click", (event) => {
          event.stopPropagation();
          handleNodeClick(node.id, event);
        });

        graphGroup.appendChild(nodeGroup);
      });

      const bbox = graphGroup.getBBox();
      const pad = 20;
      const width = Math.ceil(bbox.width + (pad * 2));
      const height = Math.ceil(bbox.height + (pad * 2));
      const viewBoxX = bbox.x - pad;
      const viewBoxY = bbox.y - pad;
      svg.setAttribute("viewBox", `${viewBoxX} ${viewBoxY} ${width} ${height}`);
      svg.setAttribute("data-export-viewbox-x", String(viewBoxX));
      svg.setAttribute("data-export-viewbox-y", String(viewBoxY));
      svg.setAttribute("data-export-width", String(width));
      svg.setAttribute("data-export-height", String(height));
    }

    function redrawCurrentGraph() {
      hideTooltip();
      drawGraph(currentGraphIndex);
    }

    function getRenderedSvg() {
      const container = document.getElementById("mynetwork");
      return container.querySelector("svg");
    }

    function sanitizeExportText(svgElement) {
      const textNodes = svgElement.querySelectorAll("text");
      textNodes.forEach((node) => {
        const original = String(node.textContent || "");
        const withoutHints = original
          .replace(/\\(Hover for prior\\)/gi, "")
          .replace(/\\(Hover for CPT\\)/gi, "")
          .trim();
        node.textContent = withoutHints;
      });
    }

    function buildExportSvgString(svg) {
      const clone = svg.cloneNode(true);
      const width = Number(svg.getAttribute("data-export-width") || 1);
      const height = Number(svg.getAttribute("data-export-height") || 1);
      const viewBoxX = Number(svg.getAttribute("data-export-viewbox-x") || 0);
      const viewBoxY = Number(svg.getAttribute("data-export-viewbox-y") || 0);
      sanitizeExportText(clone);
      clone.setAttribute("width", String(width));
      clone.setAttribute("height", String(height));
      clone.setAttribute("viewBox", `${viewBoxX} ${viewBoxY} ${width} ${height}`);
      clone.removeAttribute("class");
      const serializer = new XMLSerializer();
      const raw = serializer.serializeToString(clone);
      if (raw.startsWith("<?xml")) {
        return raw;
      }
      return `<?xml version="1.0" encoding="UTF-8"?>\\n${raw}`;
    }

    function downloadBlob(content, mimeType, fileName) {
      const blob = new Blob([content], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = fileName;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    }

    async function exportCurrentView() {
      const svg = getRenderedSvg();
      if (!svg) {
        alert("No graph is currently rendered.");
        return;
      }
      const format = document.getElementById("export-format").value;
      const baseName = `belief-graph-${currentGraphIndex + 1}`;
      const exportSvgString = buildExportSvgString(svg);
      if (format === "svg") {
        downloadBlob(exportSvgString, "image/svg+xml;charset=utf-8", `${baseName}.svg`);
        return;
      }
      if (!window.jspdf || !window.jspdf.jsPDF) {
        alert("jsPDF failed to load.");
        return;
      }
      const width = Number(svg.getAttribute("data-export-width") || 1);
      const height = Number(svg.getAttribute("data-export-height") || 1);
      const orientation = width >= height ? "landscape" : "portrait";
      const pdf = new window.jspdf.jsPDF({
        orientation: orientation,
        unit: "pt",
        format: [width, height],
      });
      const parser = new DOMParser();
      const svgDoc = parser.parseFromString(exportSvgString, "image/svg+xml");
      const svgElement = svgDoc.documentElement;
      try {
        await pdf.svg(svgElement, { x: 0, y: 0, width: width, height: height });
        pdf.save(`${baseName}.pdf`);
      } catch (error) {
        alert(`PDF export failed: ${String(error)}`);
      }
    }

    function buildAllEditedPayloads() {
      const payloads = [];
      for (let idx = 0; idx < graphsData.length; idx += 1) {
        currentGraphIndex = idx;
        const graph = normalizeCurrentGraph();
        if (!graph || !graph.editable) {
          continue;
        }
        const payload = deepClone(graph.editable);
        payload.bayesian_network = deepClone(payload.bayesian_network || {});
        payloads.push(payload);
      }
      return payloads;
    }

    function validatePayloadsForJsonl(payloads) {
      if (!Array.isArray(payloads) || !payloads.length) {
        return "No editable graph payloads are available.";
      }
      for (let idx = 0; idx < payloads.length; idx += 1) {
        const payload = payloads[idx];
        if (!payload || typeof payload !== "object") {
          return `Invalid payload at graph ${idx + 1}.`;
        }
        const bn = payload.bayesian_network;
        if (!bn || typeof bn !== "object") {
          return `Missing bayesian_network at graph ${idx + 1}.`;
        }
        if (!Array.isArray(bn.belief_nodes)) {
          return `Missing belief_nodes at graph ${idx + 1}.`;
        }
        if (!Array.isArray(bn.edges)) {
          return `Missing edges at graph ${idx + 1}.`;
        }
        if (!Array.isArray(bn.joint_distribution)) {
          return `Missing joint_distribution at graph ${idx + 1}.`;
        }
      }
      return "";
    }

    function exportAllGraphsJsonl() {
      const oldIndex = currentGraphIndex;
      const payloads = buildAllEditedPayloads();
      currentGraphIndex = oldIndex;
      redrawCurrentGraph();

      const validationError = validatePayloadsForJsonl(payloads);
      if (validationError) {
        updateStatus(`Export blocked: ${validationError}`);
        alert(validationError);
        return;
      }

      const lines = payloads.map((payload) => JSON.stringify(payload));
      const content = `${lines.join("\\n")}\\n`;
      const fileName = "edited_bayesian_networks.jsonl";
      downloadBlob(content, "application/x-ndjson;charset=utf-8", fileName);
      updateStatus(
        `Exported ${payloads.length} edited graphs to ${fileName}. You can replace the original JSONL with this file.`,
      );
    }

    document.addEventListener("click", () => {
      hideActionMenu();
    });

    if (graphsData.length > 0) {
      populateSelector();
      switchGraph(0);
    } else {
      document.getElementById("mynetwork").innerHTML =
        "<h3 style='padding:20px'>No valid graphs found in the data file.</h3>";
    }
  </script>
</body>
</html>
"""


def format_cpt_tooltip(node_name: str, bn_data: dict) -> str:
    """Format a Conditional Probability Table as an HTML tooltip."""
    if not bn_data or node_name not in bn_data:
        return "No CPT available"

    cpt = bn_data[node_name]
    parents = cpt.get("parents", [])
    probs = cpt.get("probabilities", {})

    html = "<div style='font-family: sans-serif; font-size: 12px;'>"
    html += f"<strong>Node: {node_name}</strong><br/><br/>"

    html += "<table border='1' cellpadding='4' style='border-collapse: collapse;'>"

    if not parents:
        html += f"<tr><th>Prior P(True)</th><td>{probs.get('prior', 0):.3f}</td></tr>"
    else:
        # Header
        html += "<tr>"
        for p in parents:
            short_p = p.replace("Belief_", "B")
            html += f"<th style='background-color:#eee'>{short_p}</th>"
        html += "<th style='background-color:#eee'>P(True)</th></tr>"
        # Rows
        for cond, val in probs.items():
            html += "<tr>"
            parts = cond.split(",")
            for part in parts:
                bool_val = part.split("=")[1]
                # Color code T/F for readability
                color = "green" if bool_val == "True" else "red"
                html += (
                    f"<td style='color:{color}; text-align:center;'>{bool_val[0]}</td>"
                )
            html += f"<td>{val:.3f}</td></tr>"

    html += "</table></div>"
    return html


def calculate_edge_weight_label(
    source_name: str, target_name: str, bn_data: dict
) -> str:
    """Calculate a rough marginal influence label for edges if CPT exists."""
    if not bn_data or target_name not in bn_data:
        return ""

    cpt = bn_data[target_name]
    parents = cpt.get("parents", [])
    probs = cpt.get("probabilities", {})

    if source_name not in parents:
        return ""

    # Calculate average difference in P(Target=True) when source goes from False->True
    # marginalized across other parents
    diffs = []

    # We find matching pairs of conditions that differ only on source_name
    # Since condition keys look like "Belief_1=True,Belief_2=False"

    paired = defaultdict(dict)

    for cond, val in probs.items():
        if cond == "prior":
            continue

        parts = cond.split(",")
        # extract value of source_name
        source_val = None
        other_conds = []
        for part in parts:
            p_name, p_val = part.split("=")
            if p_name == source_name:
                source_val = p_val
            else:
                other_conds.append(part)

        other_key = ",".join(other_conds)
        paired[other_key][source_val] = val

    for other_key, vals in paired.items():
        if "True" in vals and "False" in vals:
            diffs.append(vals["True"] - vals["False"])

    if not diffs:
        return ""

    avg_diff = sum(diffs) / len(diffs)
    sign = "+" if avg_diff > 0 else ""
    return f"<b>dP: {sign}{avg_diff:.2f}</b>"


def visualize_graphs(file_path: str) -> str:
    """
    Load belief graphs from a JSONL file and return an HTML string visualization.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        An HTML string containing the interactive visualization.
    """
    raw_graphs = read_jsonl_graphs(file_path, required_keys=["bayesian_network"])
    formatted_graphs = []

    for item in raw_graphs:
        graph_data = item.get("bayesian_network") or {}

        target_text = graph_data.get("target") or item.get("id") or "Target Proposition"
        bn_data = graph_data.get("bayesian_network", {})

        vis_nodes = []

        # Node 0 (Target)
        target_label = f"<b>Target:</b>\n{target_text}"
        if bn_data:
            target_label += "\n<i>(Hover for CPT)</i>"

        vis_nodes.append(
            {
                "id": 0,
                "label": target_label,
                "color": {"background": "#D2E5FF", "border": "#2B7CE9"},
                "borderWidth": 2,
                "title": format_cpt_tooltip("Target", bn_data),
            }
        )

        # Belief Nodes (1 to N)
        for i, belief_text in enumerate(graph_data.get("belief_nodes", [])):
            node_id = i + 1
            node_name = f"Belief_{node_id}"

            b_label = f"<b>B{node_id}:</b> {belief_text}"
            if bn_data:
                b_label += "\n<i>(Hover for prior)</i>"

            vis_nodes.append(
                {
                    "id": node_id,
                    "label": b_label,
                    "color": {"background": "#ffffff", "border": "#999999"},
                    "title": format_cpt_tooltip(node_name, bn_data),
                }
            )
        vis_edges = []
        for edge in graph_data.get("edges", []):
            is_pos = edge.get("positive_influence", True)

            source_idx = edge.get("from")
            target_idx = edge.get("to")

            source_name = "Target" if source_idx == 0 else f"Belief_{source_idx}"
            target_name = "Target" if target_idx == 0 else f"Belief_{target_idx}"

            # Use structure sign for base visual styling
            color = "#4CAF50" if is_pos else "#F44336"
            base_label = "+" if is_pos else "-"

            if bn_data:
                # If we have CPTs, calculate empirical influence difference
                cpt_label = calculate_edge_weight_label(
                    source_name, target_name, bn_data
                )
                label = cpt_label if cpt_label else f"<b>{base_label}</b>"
            else:
                label = f"<b>{base_label}</b>"

            vis_edges.append(
                {
                    "from": source_idx,
                    "to": target_idx,
                    "color": {"color": color},
                    "label": label,
                    "font": {"background": "white", "size": 16, "multi": "html"},
                    "dashes": not is_pos,
                    "width": 2,
                }
            )

        formatted_graphs.append(
            {
                "target": target_text,
                "nodes": vis_nodes,
                "edges": vis_edges,
                "editable": {
                    "id": item.get("id"),
                    "factual_domain": item.get("factual_domain"),
                    "proposition_is_correct": item.get("proposition_is_correct"),
                    "control_dialogue": item.get("control_dialogue"),
                    "original_text": item.get("original_text"),
                    "proposition_source": item.get("proposition_source"),
                    "bayesian_network": graph_data,
                },
            }
        )

    if not formatted_graphs:
        return ""

    return HTML_TEMPLATE.replace("__GRAPH_DATA__", json.dumps(formatted_graphs))


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Visualize belief graphs as HTML.")
    parser.add_argument(
        "--input",
        type=str,
        default="src/simulation/data/belief_structures.jsonl",
        help="Input JSONL file containing graphs (or fitted BN files)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="graph_visualization.html",
        help="Output HTML file",
    )
    args = parser.parse_args()

    html_content = visualize_graphs(args.input)

    if not html_content:
        print(f"Error: No valid graphs found in {args.input}")
        return

    with open(args.output, "w", encoding="utf-8") as f_out:
        f_out.write(html_content)

    print("Successfully generated visualization.")
    print(f"Open '{os.path.abspath(args.output)}' in your web browser to view it.")


if __name__ == "__main__":
    main()
