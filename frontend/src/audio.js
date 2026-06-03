// src/audio.js
let audioCtx = null;

// hold the warmed MediaStream here
let _warmStream = null;

/**
 * Call once in a true user-gesture (e.g.  "Allow Microphone" click)
 * so that Firefox's pipeline gets going.
 */
export async function warmupMicrophone() {
  if (_warmStream) return _warmStream;
  _warmStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  // disable tracks until you’re ready to record
  _warmStream.getTracks().forEach((t) => (t.enabled = false));
  return _warmStream;
}

/**
 * When ready to record, pull back out the stream
 */
export function getWarmStream() {
  return _warmStream;
}

/**
 * Call once in a true user-gesture (e.g. your “Play Tone” button click)
 * so that the AudioContext can move to running state.
 */
export function getAudioContext() {
  if (!audioCtx) {
    const Ctor = window.AudioContext || window.webkitAudioContext;
    audioCtx = new Ctor();
  }
  return audioCtx;
}

/**
 * Call in a user‐gesture to unlock the context.
 */
export function audioContextSetup() {
  return audioCtx !== null;
}

/**
 * Decode a Base64‐encoded payload and play it to completion.
 * Resolves when playback ends.
 */
export async function playBase64Audio(dataUrl) {
  // grab the ArrayBuffer of the data: URL
  //    fetch() knows how to parse data: URLs
  const resp = await fetch(dataUrl);
  const arrayBuffer = await resp.arrayBuffer();

  // decode it
  const audioBuffer = await new Promise((resolve, reject) => {
    // newer browsers return a promise:
    const p = audioCtx.decodeAudioData(arrayBuffer);
    if (p && p.then) {
      p.then(resolve, reject);
    } else {
      // fallback to old callback style
      audioCtx.decodeAudioData(arrayBuffer, resolve, reject);
    }
  });

  // create a source, wire it up, and play
  const source = audioCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(audioCtx.destination);

  // return a promise that resolves when it ends
  return new Promise((resolve) => {
    source.onended = resolve;
    source.start(0);
  });
}
