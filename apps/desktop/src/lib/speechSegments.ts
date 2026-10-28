/** Splits Aya's streamed answer into speakable segments.
 *
 * Cuts land only where the backend's per-segment cleaning still works: never
 * inside code (fenced or inline), math, a link target or a bare URL, and a
 * period only ends a sentence when whitespace follows it (so "63.2%" stays one
 * number and a trailing "." waits for the next delta). */

export type FlushMode = "stream" | "idle" | "final";

const MIN_CHARS = 20;
// The answer's first segment may end at a comma once it is this long, so the
// voice starts before the first full sentence has streamed in.
const FIRST_MIN_CHARS = 12;
const MAX_CHARS = 160;
// A single "$" opens math only when it is closed soon after; otherwise it is a
// currency sign or a stray character and must not freeze the cutter.
const MATH_LOOKAHEAD = 200;

/** protectedMask[i] is true when character i belongs to a span that must not be
 * split. Spans still open at the end run to the end of the text. */
export function protectedMask(text: string): boolean[] {
  const mask = new Array<boolean>(text.length).fill(false);
  const mark = (start: number, end: number) => {
    for (let i = start; i < end; i++) mask[i] = true;
  };
  let i = 0;
  while (i < text.length) {
    if (text.startsWith("```", i)) {
      const close = text.indexOf("```", i + 3);
      const end = close < 0 ? text.length : close + 3;
      mark(i, end);
      i = end;
    } else if (text[i] === "`") {
      const close = text.indexOf("`", i + 1);
      const end = close < 0 ? text.length : close + 1;
      mark(i, end);
      i = end;
    } else if (text.startsWith("$$", i)) {
      const close = text.indexOf("$$", i + 2);
      const end = close < 0 ? text.length : close + 2;
      mark(i, end);
      i = end;
    } else if (text[i] === "$" && /[^\s\d$]/.test(text[i + 1] ?? "")) {
      const close = text.indexOf("$", i + 1);
      if (close >= 0 && close - i <= MATH_LOOKAHEAD) {
        mark(i, close + 1);
        i = close + 1;
      } else if (close < 0 && text.length - i <= MATH_LOOKAHEAD) {
        mark(i, text.length);
        i = text.length;
      } else {
        i++;
      }
    } else if (text.startsWith("](", i)) {
      const close = text.indexOf(")", i + 2);
      const end = close < 0 ? text.length : close + 1;
      mark(i, end);
      i = end;
    } else if (/^(https?:\/\/|www\.)/i.test(text.slice(i, i + 8))) {
      let end = i;
      while (end < text.length && !/\s/.test(text[end])) end++;
      mark(i, end);
      i = end;
    } else {
      i++;
    }
  }
  return mask;
}

const SENTENCE_END = /[.!?…]/;
const CJK_SENTENCE_END = /[。！？]/;

function findCut(text: string, mask: boolean[], mode: FlushMode, first: boolean): number {
  const free = (i: number) => i >= 0 && i < text.length && !mask[i];
  if (first) {
    for (let i = 0; i < text.length && i < MAX_CHARS; i++) {
      if (!free(i) || !/[,;:、，]/.test(text[i]) || !/\s/.test(text[i + 1] ?? "")) continue;
      if (/\d/.test(text[i - 1] ?? "") && /\d/.test(text[i + 2] ?? "")) continue; // "2, 3" lists of numbers read better whole
      if (text.slice(0, i + 1).trim().length >= FIRST_MIN_CHARS) return i + 1;
    }
  }
  // 1. Sentence ends (and line breaks) once a segment is long enough.
  for (let i = 0; i < text.length && i < MAX_CHARS; i++) {
    if (!free(i)) continue;
    const c = text[i];
    const next = text[i + 1];
    const isEnd =
      c === "\n" ||
      CJK_SENTENCE_END.test(c) ||
      (SENTENCE_END.test(c) && next !== undefined && /\s/.test(next));
    if (isEnd && text.slice(0, i + 1).trim().length >= MIN_CHARS) return i + 1;
  }
  // 2. Too long without a sentence end: cut at the last free comma or space.
  if (text.length > MAX_CHARS) {
    let space = -1;
    for (let i = Math.min(MAX_CHARS, text.length) - 1; i > 0; i--) {
      if (!free(i) || !/\s/.test(text[i])) continue;
      if (text[i - 1] === "," || text[i - 1] === ";") return i + 1;
      if (space < 0) space = i;
    }
    if (space > 0) return space + 1;
    // A protected span crosses the limit: cut after it, at the next free space.
    for (let i = MAX_CHARS; i < text.length; i++) {
      if (free(i) && /\s/.test(text[i])) return i + 1;
    }
  }
  // 3. The stream paused: speak what is safely complete (up to the last free space).
  if (mode === "idle") {
    for (let i = text.length - 1; i > 0; i--) {
      if (free(i) && /\s/.test(text[i]) && /\p{L}/u.test(text.slice(0, i))) return i + 1;
    }
  }
  return 0;
}

function withoutCode(segment: string): string {
  return segment.replace(/```[\s\S]*?```/g, " ").trim();
}

/** Incremental segmentation: returns the segments ready to speak and the text
 * to keep for the next call. "final" (end of the answer) emits everything except
 * an unclosed code fence. */
export function segmentSpeech(
  text: string, mode: FlushMode = "stream", first = false,
): { segments: string[]; remainder: string } {
  const segments: string[] = [];
  const emit = (segment: string) => {
    const clean = withoutCode(segment);
    if (/[\p{L}\p{N}]/u.test(clean)) segments.push(clean);
  };
  for (;;) {
    const mask = protectedMask(text);
    const cut = findCut(text, mask, mode, first && segments.length === 0);
    if (!cut) break;
    emit(text.slice(0, cut));
    text = text.slice(cut);
  }
  if (mode === "final") {
    const fences = text.split("```").length - 1;
    emit(fences % 2 === 1 ? text.slice(0, text.lastIndexOf("```")) : text);
    text = "";
  }
  return { segments, remainder: text };
}
