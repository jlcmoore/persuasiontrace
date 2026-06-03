// Helper functions for reflection highlight range calculations.

// Compute start/end offsets of the selection relative to container text.
export function getSelectionOffsets(container, range) {
  const preSelectionRange = range.cloneRange();
  preSelectionRange.selectNodeContents(container);
  preSelectionRange.setEnd(range.startContainer, range.startOffset);
  const start = preSelectionRange.toString().length;
  const selectedText = range.toString();
  const end = start + selectedText.length;
  return { start, end };
}

// Clamp a range to the length of the message text.
export function normalizeRange(range, textLength) {
  if (!range) {
    return null;
  }
  const rawStart = Number.isFinite(range.start) ? range.start : 0;
  const rawEnd = Number.isFinite(range.end) ? range.end : textLength;
  const start = Math.max(0, Math.min(rawStart ?? 0, textLength));
  const end = Math.max(start, Math.min(rawEnd ?? start, textLength));
  if (end <= start) {
    return null;
  }
  return { start, end };
}

// Normalize and merge highlight ranges for a given length.
export function normalizeRangesForLength(ranges, textLength) {
  if (!Array.isArray(ranges) || ranges.length === 0) {
    return [];
  }
  const normalized = ranges
    .map((range) => normalizeRange(range, textLength))
    .filter((range) => range !== null)
    .sort((a, b) => a.start - b.start);
  if (normalized.length === 0) {
    return [];
  }
  const merged = [{ ...normalized[0] }];
  for (let i = 1; i < normalized.length; i += 1) {
    const current = normalized[i];
    const last = merged[merged.length - 1];
    if (current.start <= last.end) {
      last.end = Math.max(last.end, current.end);
    } else {
      merged.push({ ...current });
    }
  }
  return merged;
}

// Add a highlight range and merge it with existing ranges.
export function addRangeToList(ranges, range, textLength) {
  const combined = [...ranges, range];
  return normalizeRangesForLength(combined, textLength);
}

// Remove a highlight segment from the existing ranges.
export function subtractRangeList(ranges, removal) {
  if (ranges.length === 0) {
    return [];
  }
  const result = [];
  ranges.forEach((range) => {
    if (removal.end <= range.start || removal.start >= range.end) {
      result.push(range);
      return;
    }
    if (removal.start > range.start) {
      result.push({
        start: range.start,
        end: Math.min(removal.start, range.end),
      });
    }
    if (removal.end < range.end) {
      result.push({
        start: Math.max(removal.end, range.start),
        end: range.end,
      });
    }
  });
  return result;
}

// Determine if a proposed range is already fully highlighted.
export function isRangeFullyCovered(ranges, target) {
  if (!Array.isArray(ranges) || ranges.length === 0) {
    return false;
  }
  let coveredUntil = target.start;
  for (const range of ranges) {
    if (range.end <= coveredUntil) {
      continue;
    }
    if (range.start > coveredUntil) {
      return false;
    }
    coveredUntil = Math.max(coveredUntil, range.end);
    if (coveredUntil >= target.end) {
      return true;
    }
  }
  return false;
}

// Split message text into highlighted and unhighlighted segments.
export function buildTextSegments(text, ranges) {
  if (!text) {
    return [];
  }
  const normalized = normalizeRangesForLength(
    Array.isArray(ranges) ? ranges : ranges ? [ranges] : [],
    text.length,
  );
  if (normalized.length === 0) {
    return [{ text, highlighted: false }];
  }
  const segments = [];
  let cursor = 0;
  normalized.forEach((range) => {
    if (range.start > cursor) {
      segments.push({
        text: text.slice(cursor, range.start),
        highlighted: false,
      });
    }
    segments.push({
      text: text.slice(range.start, range.end),
      highlighted: true,
    });
    cursor = range.end;
  });
  if (cursor < text.length) {
    segments.push({
      text: text.slice(cursor),
      highlighted: false,
    });
  }
  return segments.filter((segment) => segment.text.length > 0);
}

// Normalize a highlight entry and attach the text snippet.
export function normalizeHighlight(entry, text) {
  const normalized = normalizeRange(entry, text.length);
  if (!normalized) {
    return null;
  }
  const snippet = text.slice(normalized.start, normalized.end);
  if (!snippet) {
    return null;
  }
  return {
    messageIndex: entry.messageIndex,
    start: normalized.start,
    end: normalized.end,
    text: snippet,
  };
}
