# 05 — Presence: avatar and voice

## Done (2026-10-25)

- Live2D Mao on a fluid stage (zoom/pan, hide/show, remembered), expressions per
  state, idle gestures/specials/dozing, poke reactions.
- Voice sidecar (VieNeu v3 Turbo + VOICEVOX): 25 Vietnamese, 13 Vietnamese-anime
  (cloned timbre), 136 Japanese voices; picker with speed/pitch/preview/credit.
- While an answer is written Aya looks at the newest line; while it is spoken
  her gaze follows the part the voice has reached; mouth from the audio level;
  small nod with the voice; the end reaction waits for the voice (immediately
  when voice is off); cut-short voice never leaves her talking.

## Next

1. **Faster voice start.** Cache the clone speaker profiles (computed once at
   setup) instead of re-enrolling 13 references at start (8.7 s → ~2 s); load
   VOICEVOX models lazily (already) and VieNeu first so `vi`/`vi-anime` are ready first.
2. **Emotion.** The answer model adds an optional mood per sentence (`[vui]`,
   `[ngạc nhiên]`, `[buồn]`, `[suy nghĩ]`) or a cheap classifier (luna) tags it;
   the stage maps mood → Mao expression + gesture, and VieNeu emotion cues
   (`[cười]`, `[thở dài]`) where natural. Tags never appear in the chat text.
3. **Speaking motion.** Subtle body sway and occasional TapBody gesture while
   speaking long answers; eye smile on happy moods; blink rate up when thinking.
4. **Interruptions.** Typing, clicking the stage or a new message stops the
   voice at the sentence boundary; "Aya, dừng lại" by voice later.
5. **Voice input (push-to-talk).** Hold a key or the mic button → local STT →
   message. STT: sherpa-onnx Zipformer Vietnamese (as in the user's RAG-TTS,
   ~4 s full pipeline there) or Whisper via MLX; streaming partial transcript.
   Then a wake word ("Aya ơi") as an opt-in.
6. **Spoken notifications.** Reminders and task completions spoken when voice
   is on and the app is focused; otherwise macOS notification as today.
7. **Japanese mode polish.** Bubble shows the Japanese line with the Vietnamese
   under it; translation cached per sentence (exists) and done ahead while the
   previous sentence plays.
8. **Characters.** Keep Mao default; allow other Live2D models the user owns
   (drop folder + license note), per-model expression map.

## Gates

Voice ready ≤ 3 s after backend start; first audio ≤ 1.5 s after the first
sentence is written (vi voices); no visible desync between mouth and audio; user
acceptance session on mood mapping.
