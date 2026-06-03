// src/utils.js

/**
 * Throws in Jest (NODE_ENV==='test' or JEST_WORKER_ID set),
 * warns in dev, no-ops in prod.
 */
export function testAssert(condition, message) {
  const isTest =
    typeof process !== "undefined" &&
    (process.env.NODE_ENV === "test" || process.env.JEST_WORKER_ID);

  if (!condition) {
    if (isTest) {
      throw new Error(message || "Test assertion failed");
    }
    console.warn("[Dev Warning]", message || "Test assertion failed");
    // in production, do nothing
  }
}

// Generates a fake participant ID
export function generateFakeID() {
  return (
    Math.random().toString(36).substring(2, 15) +
    Math.random().toString(36).substring(2, 15)
  );
}

// Formats a seconds value for display
export function formatMMSS(totalSeconds) {
  const s = totalSeconds || 0;
  const m = Math.floor(s / 60);
  const sec = s % 60;
  // pad to two digits
  const MM = String(m).padStart(2, "0");
  const SS = String(sec).padStart(2, "0");
  return `${MM}:${SS}`;
}
