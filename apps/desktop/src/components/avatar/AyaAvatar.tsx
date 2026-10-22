import { memo } from "react";
import type { AvatarState } from "../../lib/types";

/**
 * Aya's on-screen character (step 1: hand-drawn SVG). Every renderer — this
 * one, a VRM model later — takes the same props, so the chat only ever says
 * which state Aya is in.
 */
export interface AvatarProps {
  state: AvatarState;
  size?: number;
}

const STATE_LABEL: Record<AvatarState, string> = {
  idle: "Aya đang chờ cậu~",
  thinking: "Aya đang suy nghĩ…",
  tool: "Aya đang làm việc…",
  waiting: "Aya đang chờ cậu đồng ý",
  talking: "Aya đang trả lời",
  happy: "Xong rồi nè~",
  confused: "Ơ… có lỗi rồi",
};

export const AyaAvatar = memo(function AyaAvatar({ state, size = 148 }: AvatarProps) {
  return (
    <div className={`aya-avatar aya-avatar--${state}`} style={{ width: size, height: size }} role="img"
      aria-label={STATE_LABEL[state]} title={STATE_LABEL[state]}>
      <svg viewBox="0 0 200 200" width={size} height={size}>
        <defs>
          <radialGradient id="aya-aura" cx="50%" cy="45%" r="55%">
            <stop offset="0%" stopColor="#fff4fb" />
            <stop offset="100%" stopColor="#f3e8ff" stopOpacity="0" />
          </radialGradient>
          <linearGradient id="aya-hair" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#c9a7ff" />
            <stop offset="100%" stopColor="#ff9fcf" />
          </linearGradient>
          <radialGradient id="aya-iris" cx="50%" cy="35%" r="70%">
            <stop offset="0%" stopColor="#b9a3ff" />
            <stop offset="60%" stopColor="#6c4bff" />
            <stop offset="100%" stopColor="#3b2391" />
          </radialGradient>
        </defs>
        <circle cx="100" cy="100" r="96" fill="url(#aya-aura)" />
        <g className="aya-body">
          {/* back hair */}
          <path d="M44 92 C40 40 160 40 156 92 L162 168 C140 176 124 160 120 140 L80 140 C76 160 60 176 38 168 Z"
            fill="url(#aya-hair)" />
          {/* shoulders */}
          <path d="M62 170 C70 150 130 150 138 170 L138 200 L62 200 Z" fill="#fdfbff" stroke="#e4d6ff" strokeWidth="2" />
          <path d="M92 156 L100 168 L108 156" fill="none" stroke="#ff8fc4" strokeWidth="3" strokeLinecap="round" />
          {/* face */}
          <ellipse cx="100" cy="102" rx="44" ry="48" fill="#ffeadb" />
          {/* blush */}
          <ellipse className="aya-blush" cx="72" cy="118" rx="9" ry="4.5" fill="#ffb3c8" />
          <ellipse className="aya-blush" cx="128" cy="118" rx="9" ry="4.5" fill="#ffb3c8" />
          {/* eyes */}
          <g className="aya-eyes">
            <g className="aya-eyes-open">
              {[78, 122].map((cx) => (
                <g key={cx} className="aya-eye">
                  <ellipse cx={cx} cy="104" rx="10.5" ry="13" fill="#fff" />
                  <ellipse className="aya-iris" cx={cx} cy="105" rx="8.5" ry="11" fill="url(#aya-iris)" />
                  <circle className="aya-iris" cx={cx} cy="106" r="3.6" fill="#24135f" />
                  <circle cx={cx + 3} cy="99" r="3.2" fill="#fff" />
                  <circle cx={cx - 3} cy="110" r="1.6" fill="#fff" opacity="0.8" />
                  <path d={`M${cx - 12} 92 Q${cx} 86 ${cx + 12} 92`} stroke="#4a2c6b" strokeWidth="3" fill="none"
                    strokeLinecap="round" />
                </g>
              ))}
            </g>
            <g className="aya-eyes-happy">
              <path d="M68 106 Q78 94 88 106" stroke="#4a2c6b" strokeWidth="3.5" fill="none" strokeLinecap="round" />
              <path d="M112 106 Q122 94 132 106" stroke="#4a2c6b" strokeWidth="3.5" fill="none" strokeLinecap="round" />
            </g>
            <g className="aya-eyes-confused">
              <path d="M70 98 L86 110 M86 98 L70 110" stroke="#4a2c6b" strokeWidth="3.5" strokeLinecap="round" />
              <path d="M114 104 Q122 96 130 104 Q122 112 116 106" stroke="#4a2c6b" strokeWidth="3" fill="none"
                strokeLinecap="round" />
            </g>
          </g>
          {/* mouth */}
          <path className="aya-mouth aya-mouth--smile" d="M92 128 Q100 135 108 128" stroke="#c4506f" strokeWidth="3"
            fill="none" strokeLinecap="round" />
          <ellipse className="aya-mouth aya-mouth--talk" cx="100" cy="130" rx="6" ry="4.5" fill="#c4506f" />
          <path className="aya-mouth aya-mouth--open" d="M90 126 Q100 142 110 126 Z" fill="#c4506f" />
          <path className="aya-mouth aya-mouth--wavy" d="M90 130 Q95 126 100 130 Q105 134 110 130" stroke="#c4506f"
            strokeWidth="3" fill="none" strokeLinecap="round" />
          {/* bangs and side locks */}
          <path d="M54 92 C54 52 146 52 146 92 C134 78 124 74 116 86 C110 72 98 70 92 86 C84 74 70 78 54 92 Z"
            fill="url(#aya-hair)" />
          <path d="M56 90 C48 120 52 146 60 160 C64 140 62 116 66 96 Z" fill="url(#aya-hair)" />
          <path d="M144 90 C152 120 148 146 140 160 C136 140 138 116 134 96 Z" fill="url(#aya-hair)" />
          {/* ahoge */}
          <path className="aya-ahoge" d="M100 58 C96 40 112 34 108 22 C120 36 108 46 104 58 Z" fill="#c9a7ff" />
          {/* star hair clip */}
          <path d="M136 66 l3.5 7 7.6 1.1 -5.5 5.4 1.3 7.6 -6.9-3.6 -6.9 3.6 1.3-7.6 -5.5-5.4 7.6-1.1 z" fill="#ffd166"
            stroke="#f4a940" strokeWidth="1.2" />
        </g>
        {/* state props */}
        <g className="aya-prop aya-prop--thinking">
          <circle cx="160" cy="52" r="16" fill="#fff" stroke="#d9c8ff" strokeWidth="2" />
          <circle cx="152" cy="52" r="2.6" fill="#8b6cff" className="aya-dot aya-dot--1" />
          <circle cx="160" cy="52" r="2.6" fill="#8b6cff" className="aya-dot aya-dot--2" />
          <circle cx="168" cy="52" r="2.6" fill="#8b6cff" className="aya-dot aya-dot--3" />
          <circle cx="142" cy="72" r="4" fill="#fff" stroke="#d9c8ff" strokeWidth="2" />
        </g>
        <g className="aya-prop aya-prop--tool">
          <path d="M162 40 l3 9 9 3 -9 3 -3 9 -3-9 -9-3 9-3 z" fill="#ffd166" />
          <path d="M40 54 l2 6 6 2 -6 2 -2 6 -2-6 -6-2 6-2 z" fill="#ff9fcf" />
        </g>
        <g className="aya-prop aya-prop--waiting">
          <circle cx="160" cy="52" r="16" fill="#fff" stroke="#ffc6e0" strokeWidth="2" />
          <text x="160" y="59" textAnchor="middle" fontSize="20" fontWeight="700" fill="#ff5fa2">?</text>
        </g>
        <g className="aya-prop aya-prop--happy">
          <path d="M156 46 c0-6 9-6 9 0 c0-6 9-6 9 0 c0 6-9 11-9 13 c0-2-9-7-9-13 z" fill="#ff7eb6" />
        </g>
        <g className="aya-prop aya-prop--confused">
          <path d="M148 70 C142 82 154 86 154 78 C154 74 150 72 148 70 Z" fill="#9fd8ff" stroke="#5bb7f0" strokeWidth="1.2" />
        </g>
      </svg>
    </div>
  );
});
