import type { TabMeasure, TimeSignature } from './domain.js'

export type RhythmGroup =
  | 'metronome'
  | 'rock'
  | 'pop'
  | 'funk'
  | 'blues'
  | 'electronic'
  | 'world'
  | 'other-meter'

export type RhythmKit =
  | 'click'
  | 'acoustic'
  | 'electronic'
  | 'brush'
  | 'wood'

export type PercussionVoice =
  | 'click-high'
  | 'click-low'
  | 'kick'
  | 'snare'
  | 'closed-hat'
  | 'open-hat'
  | 'crash'
  | 'clap'
  | 'rim'
  | 'tom-high'
  | 'tom-low'
  | 'shaker'
  | 'cowbell'

export interface RhythmHit {
  step: number
  voice: PercussionVoice
  velocity: number
}

export interface PracticeRhythm {
  id: string
  label: string
  group: RhythmGroup
  kit: RhythmKit
  timeSignature: TimeSignature
  stepsPerBar: number
  bpmRange: [number, number]
  swing: number
  hits: RhythmHit[]
}

export interface RhythmCompatibility {
  compatible: boolean
  reason?: 'time-signature' | 'tempo-low' | 'tempo-high'
  message?: string
}

export interface PracticeTempoProfile {
  low: number
  typical: number
  high: number
}

export const PRACTICE_OUTPUT_GAIN = 2

const FOUR_FOUR: TimeSignature = [4, 4]
const THREE_FOUR: TimeSignature = [3, 4]
const SIX_EIGHT: TimeSignature = [6, 8]
const TWO_FOUR: TimeSignature = [2, 4]

function hits(
  voice: PercussionVoice,
  steps: number[],
  velocity = 0.78,
): RhythmHit[] {
  return steps.map((step) => ({ step, voice, velocity }))
}

function every(
  voice: PercussionVoice,
  stepCount: number,
  spacing: number,
  velocity = 0.68,
  offset = 0,
): RhythmHit[] {
  const steps: number[] = []
  for (let step = offset; step < stepCount; step += spacing) steps.push(step)
  return hits(voice, steps, velocity)
}

function rhythm(
  definition: Omit<PracticeRhythm, 'swing'> & { swing?: number },
): PracticeRhythm {
  return {
    ...definition,
    swing: definition.swing ?? 0.5,
  }
}

export const PRACTICE_RHYTHMS: PracticeRhythm[] = [
  rhythm({
    id: 'click-quarter',
    label: '经典四分音符',
    group: 'metronome',
    kit: 'click',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 4,
    bpmRange: [30, 260],
    hits: [
      ...hits('click-high', [0], 1),
      ...hits('click-low', [1, 2, 3], 0.72),
    ],
  }),
  rhythm({
    id: 'click-eighth',
    label: '八分音符细分',
    group: 'metronome',
    kit: 'click',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 8,
    bpmRange: [30, 220],
    hits: [
      ...hits('click-high', [0], 1),
      ...hits('click-low', [2, 4, 6], 0.74),
      ...hits('click-low', [1, 3, 5, 7], 0.42),
    ],
  }),
  rhythm({
    id: 'click-sixteenth',
    label: '十六分音符细分',
    group: 'metronome',
    kit: 'click',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [30, 150],
    hits: [
      ...hits('click-high', [0], 1),
      ...hits('click-low', [4, 8, 12], 0.76),
      ...hits('click-low', [2, 6, 10, 14], 0.48),
      ...hits('click-low', [1, 3, 5, 7, 9, 11, 13, 15], 0.28),
    ],
  }),
  rhythm({
    id: 'click-triplet',
    label: '三连音细分',
    group: 'metronome',
    kit: 'wood',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 12,
    bpmRange: [35, 190],
    hits: [
      ...hits('click-high', [0], 1),
      ...hits('click-low', [3, 6, 9], 0.76),
      ...hits('click-low', [1, 2, 4, 5, 7, 8, 10, 11], 0.36),
    ],
  }),
  rhythm({
    id: 'wood-quarter',
    label: '木鱼强拍',
    group: 'metronome',
    kit: 'wood',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 4,
    bpmRange: [30, 240],
    hits: [
      ...hits('cowbell', [0], 0.92),
      ...hits('rim', [1, 2, 3], 0.58),
    ],
  }),
  rhythm({
    id: 'shaker-eighth',
    label: '沙锤八分律动',
    group: 'metronome',
    kit: 'acoustic',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 8,
    bpmRange: [45, 190],
    hits: [
      ...every('shaker', 8, 1, 0.46),
      ...hits('shaker', [0, 4], 0.78),
    ],
  }),
  rhythm({
    id: 'rock-basic',
    label: '基础摇滚',
    group: 'rock',
    kit: 'acoustic',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [55, 180],
    hits: [
      ...hits('kick', [0, 8], 0.92),
      ...hits('snare', [4, 12], 0.9),
      ...every('closed-hat', 16, 2, 0.55),
    ],
  }),
  rhythm({
    id: 'rock-driving',
    label: '推进摇滚',
    group: 'rock',
    kit: 'acoustic',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [80, 190],
    hits: [
      ...hits('kick', [0, 3, 8, 10], 0.9),
      ...hits('snare', [4, 12], 0.94),
      ...every('closed-hat', 16, 2, 0.62),
      ...hits('open-hat', [14], 0.56),
    ],
  }),
  rhythm({
    id: 'rock-half-time',
    label: '半拍重摇滚',
    group: 'rock',
    kit: 'acoustic',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [65, 150],
    hits: [
      ...hits('kick', [0, 7, 10], 0.96),
      ...hits('snare', [8], 1),
      ...every('closed-hat', 16, 2, 0.5),
    ],
  }),
  rhythm({
    id: 'rock-double-time',
    label: '双倍速摇滚',
    group: 'rock',
    kit: 'acoustic',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [105, 220],
    hits: [
      ...hits('kick', [0, 6, 8, 14], 0.9),
      ...hits('snare', [2, 6, 10, 14], 0.84),
      ...every('closed-hat', 16, 1, 0.48),
    ],
  }),
  rhythm({
    id: 'rock-tom',
    label: '落地鼓推进',
    group: 'rock',
    kit: 'acoustic',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [60, 145],
    hits: [
      ...hits('kick', [0, 8], 0.94),
      ...hits('snare', [4, 12], 0.82),
      ...hits('tom-low', [2, 6, 10, 14], 0.56),
      ...hits('tom-high', [15], 0.7),
    ],
  }),
  rhythm({
    id: 'punk-fast',
    label: '快速朋克',
    group: 'rock',
    kit: 'acoustic',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [140, 240],
    hits: [
      ...hits('crash', [0], 0.72),
      ...hits('kick', [0, 4, 8, 12], 0.9),
      ...hits('snare', [2, 6, 10, 14], 0.9),
      ...every('closed-hat', 16, 2, 0.56),
      ...hits('open-hat', [3, 7, 11, 15], 0.42),
    ],
  }),
  rhythm({
    id: 'rock-arena-fill',
    label: '体育场摇滚加花',
    group: 'rock',
    kit: 'acoustic',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [70, 165],
    hits: [
      ...hits('crash', [0], 0.84),
      ...hits('kick', [0, 7, 8], 0.94),
      ...hits('snare', [4, 12, 15], 0.9),
      ...hits('closed-hat', [0, 2, 4, 6, 8, 10], 0.52),
      ...hits('tom-high', [13], 0.72),
      ...hits('tom-low', [14], 0.82),
    ],
  }),
  rhythm({
    id: 'rock-progressive-fill',
    label: '渐进摇滚轮鼓',
    group: 'rock',
    kit: 'acoustic',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [65, 150],
    hits: [
      ...hits('crash', [0], 0.78),
      ...hits('kick', [0, 3, 6, 8, 10], 0.9),
      ...hits('snare', [4, 12], 0.92),
      ...hits('closed-hat', [0, 2, 4, 6, 8, 10], 0.46),
      ...hits('tom-high', [11, 13], 0.68),
      ...hits('tom-low', [14, 15], 0.8),
    ],
  }),
  rhythm({
    id: 'pop-clean',
    label: '清爽流行',
    group: 'pop',
    kit: 'acoustic',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [65, 150],
    hits: [
      ...hits('kick', [0, 6, 8, 11], 0.86),
      ...hits('snare', [4, 12], 0.82),
      ...every('closed-hat', 16, 2, 0.42),
      ...hits('open-hat', [14], 0.5),
    ],
  }),
  rhythm({
    id: 'pop-four-floor',
    label: '四踩流行',
    group: 'pop',
    kit: 'electronic',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [90, 145],
    hits: [
      ...hits('kick', [0, 4, 8, 12], 0.86),
      ...hits('clap', [4, 12], 0.72),
      ...hits('open-hat', [2, 6, 10, 14], 0.5),
    ],
  }),
  rhythm({
    id: 'pop-syncopated',
    label: '切分流行',
    group: 'pop',
    kit: 'electronic',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [72, 132],
    hits: [
      ...hits('kick', [0, 6, 9], 0.88),
      ...hits('clap', [4, 12], 0.78),
      ...every('closed-hat', 16, 2, 0.48),
      ...hits('open-hat', [11], 0.52),
    ],
  }),
  rhythm({
    id: 'pop-acoustic',
    label: '原声手鼓',
    group: 'pop',
    kit: 'wood',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [55, 135],
    hits: [
      ...hits('kick', [0, 8], 0.64),
      ...hits('rim', [4, 12], 0.82),
      ...every('shaker', 16, 2, 0.44),
    ],
  }),
  rhythm({
    id: 'pop-transition-fill',
    label: '流行过门加花',
    group: 'pop',
    kit: 'acoustic',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [72, 150],
    hits: [
      ...hits('crash', [0], 0.68),
      ...hits('kick', [0, 7, 8], 0.84),
      ...hits('snare', [4, 12], 0.84),
      ...every('closed-hat', 16, 2, 0.44),
      ...hits('open-hat', [10], 0.5),
      ...hits('tom-high', [13, 14], 0.62),
      ...hits('tom-low', [15], 0.76),
    ],
  }),
  rhythm({
    id: 'funk-sixteenth',
    label: '十六分放克',
    group: 'funk',
    kit: 'acoustic',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [70, 125],
    hits: [
      ...hits('kick', [0, 3, 7, 10], 0.86),
      ...hits('snare', [4, 12, 15], 0.84),
      ...every('closed-hat', 16, 1, 0.4),
      ...hits('open-hat', [6, 14], 0.56),
    ],
  }),
  rhythm({
    id: 'funk-pocket',
    label: '口袋律动',
    group: 'funk',
    kit: 'acoustic',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [62, 112],
    hits: [
      ...hits('kick', [0, 5, 10, 14], 0.84),
      ...hits('snare', [4, 12], 0.92),
      ...hits('rim', [7, 15], 0.42),
      ...every('closed-hat', 16, 2, 0.5),
    ],
  }),
  rhythm({
    id: 'funk-ghost-fill',
    label: '放克幽灵音加花',
    group: 'funk',
    kit: 'acoustic',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [65, 128],
    hits: [
      ...hits('kick', [0, 3, 7, 10], 0.84),
      ...hits('snare', [4, 12], 0.9),
      ...hits('rim', [2, 6, 11, 14], 0.36),
      ...every('closed-hat', 16, 1, 0.34),
      ...hits('open-hat', [6, 14], 0.5),
      ...hits('tom-high', [15], 0.62),
    ],
  }),
  rhythm({
    id: 'neo-soul',
    label: 'Neo Soul 后拍',
    group: 'funk',
    kit: 'brush',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [55, 105],
    swing: 0.57,
    hits: [
      ...hits('kick', [0, 6, 11], 0.72),
      ...hits('snare', [4, 12], 0.7),
      ...every('closed-hat', 16, 2, 0.38),
    ],
  }),
  rhythm({
    id: 'disco',
    label: '迪斯科',
    group: 'funk',
    kit: 'electronic',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [105, 138],
    hits: [
      ...hits('kick', [0, 4, 8, 12], 0.9),
      ...hits('clap', [4, 12], 0.84),
      ...hits('open-hat', [2, 6, 10, 14], 0.62),
      ...every('closed-hat', 16, 2, 0.34),
    ],
  }),
  rhythm({
    id: 'motown',
    label: 'Motown',
    group: 'funk',
    kit: 'acoustic',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [80, 145],
    hits: [
      ...hits('kick', [0, 7, 10], 0.82),
      ...hits('snare', [4, 12], 0.88),
      ...hits('clap', [4, 12], 0.42),
      ...every('closed-hat', 16, 2, 0.5),
    ],
  }),
  rhythm({
    id: 'blues-shuffle',
    label: '布鲁斯 Shuffle',
    group: 'blues',
    kit: 'brush',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 12,
    bpmRange: [55, 155],
    hits: [
      ...hits('kick', [0, 6], 0.78),
      ...hits('snare', [3, 9], 0.76),
      ...hits('closed-hat', [0, 2, 3, 5, 6, 8, 9, 11], 0.48),
    ],
  }),
  rhythm({
    id: 'slow-blues',
    label: '慢速布鲁斯',
    group: 'blues',
    kit: 'brush',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 12,
    bpmRange: [35, 90],
    hits: [
      ...hits('kick', [0, 6], 0.66),
      ...hits('rim', [3, 9], 0.62),
      ...hits('shaker', [0, 2, 3, 5, 6, 8, 9, 11], 0.34),
    ],
  }),
  rhythm({
    id: 'boogie',
    label: 'Boogie',
    group: 'blues',
    kit: 'acoustic',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 12,
    bpmRange: [90, 185],
    hits: [
      ...hits('kick', [0, 5, 6, 11], 0.82),
      ...hits('snare', [3, 9], 0.86),
      ...hits('closed-hat', [0, 3, 6, 9], 0.5),
      ...hits('open-hat', [2, 5, 8, 11], 0.42),
    ],
  }),
  rhythm({
    id: 'house',
    label: 'House',
    group: 'electronic',
    kit: 'electronic',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [112, 132],
    hits: [
      ...hits('kick', [0, 4, 8, 12], 0.94),
      ...hits('clap', [4, 12], 0.82),
      ...hits('open-hat', [2, 6, 10, 14], 0.62),
      ...hits('closed-hat', [0, 4, 8, 12], 0.28),
    ],
  }),
  rhythm({
    id: 'electro',
    label: 'Electro',
    group: 'electronic',
    kit: 'electronic',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [95, 135],
    hits: [
      ...hits('kick', [0, 7, 8, 11], 0.9),
      ...hits('clap', [4, 12], 0.88),
      ...every('closed-hat', 16, 2, 0.4),
    ],
  }),
  rhythm({
    id: 'synth-pop',
    label: 'Synth Pop',
    group: 'electronic',
    kit: 'electronic',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [85, 145],
    hits: [
      ...hits('kick', [0, 8, 10], 0.82),
      ...hits('clap', [4, 12], 0.78),
      ...every('closed-hat', 16, 2, 0.42),
      ...hits('open-hat', [14], 0.54),
    ],
  }),
  rhythm({
    id: 'drum-and-bass',
    label: 'Drum & Bass',
    group: 'electronic',
    kit: 'electronic',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [150, 190],
    hits: [
      ...hits('kick', [0, 6, 10], 0.94),
      ...hits('snare', [4, 12], 0.96),
      ...every('closed-hat', 16, 1, 0.36),
    ],
  }),
  rhythm({
    id: 'drum-and-bass-fill',
    label: 'DnB 碎拍加花',
    group: 'electronic',
    kit: 'electronic',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [145, 195],
    hits: [
      ...hits('crash', [0], 0.72),
      ...hits('kick', [0, 6, 10, 15], 0.94),
      ...hits('snare', [4, 12], 0.96),
      ...every('closed-hat', 16, 1, 0.33),
      ...hits('tom-high', [13], 0.62),
      ...hits('tom-low', [14], 0.74),
    ],
  }),
  rhythm({
    id: 'trap',
    label: 'Trap 半拍',
    group: 'electronic',
    kit: 'electronic',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [65, 150],
    hits: [
      ...hits('kick', [0, 7, 11], 0.96),
      ...hits('clap', [8], 0.9),
      ...every('closed-hat', 16, 2, 0.42),
      ...hits('closed-hat', [13, 14, 15], 0.48),
    ],
  }),
  rhythm({
    id: 'bossa-nova',
    label: 'Bossa Nova',
    group: 'world',
    kit: 'brush',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [70, 155],
    hits: [
      ...hits('kick', [0, 6, 8, 14], 0.62),
      ...hits('rim', [0, 3, 6, 10, 13], 0.72),
      ...every('shaker', 16, 2, 0.34),
    ],
  }),
  rhythm({
    id: 'samba',
    label: 'Samba',
    group: 'world',
    kit: 'acoustic',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [95, 205],
    hits: [
      ...hits('kick', [0, 3, 4, 7, 8, 11, 12, 15], 0.7),
      ...hits('rim', [2, 6, 10, 14], 0.68),
      ...every('shaker', 16, 1, 0.36),
    ],
  }),
  rhythm({
    id: 'reggae',
    label: 'Reggae One Drop',
    group: 'world',
    kit: 'acoustic',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [58, 118],
    hits: [
      ...hits('kick', [8], 0.82),
      ...hits('rim', [8], 0.86),
      ...hits('closed-hat', [2, 6, 10, 14], 0.52),
    ],
  }),
  rhythm({
    id: 'ska',
    label: 'Ska',
    group: 'world',
    kit: 'acoustic',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [110, 210],
    hits: [
      ...hits('kick', [0, 8], 0.82),
      ...hits('snare', [4, 12], 0.9),
      ...hits('open-hat', [2, 6, 10, 14], 0.58),
    ],
  }),
  rhythm({
    id: 'son-clave',
    label: 'Son Clave 3-2',
    group: 'world',
    kit: 'wood',
    timeSignature: FOUR_FOUR,
    stepsPerBar: 16,
    bpmRange: [70, 190],
    hits: [
      ...hits('cowbell', [0, 3, 6, 10, 12], 0.76),
      ...hits('shaker', [2, 6, 10, 14], 0.32),
    ],
  }),
  rhythm({
    id: 'waltz-classic',
    label: '古典华尔兹',
    group: 'other-meter',
    kit: 'acoustic',
    timeSignature: THREE_FOUR,
    stepsPerBar: 12,
    bpmRange: [55, 145],
    hits: [
      ...hits('kick', [0], 0.82),
      ...hits('snare', [4, 8], 0.58),
      ...every('closed-hat', 12, 2, 0.34),
    ],
  }),
  rhythm({
    id: 'waltz-brisk',
    label: '快速华尔兹',
    group: 'other-meter',
    kit: 'brush',
    timeSignature: THREE_FOUR,
    stepsPerBar: 6,
    bpmRange: [120, 210],
    hits: [
      ...hits('kick', [0], 0.72),
      ...hits('snare', [2, 4], 0.48),
      ...every('shaker', 6, 1, 0.32),
    ],
  }),
  rhythm({
    id: 'six-eight-ballad',
    label: '6/8 抒情',
    group: 'other-meter',
    kit: 'brush',
    timeSignature: SIX_EIGHT,
    stepsPerBar: 12,
    bpmRange: [45, 115],
    hits: [
      ...hits('kick', [0, 6], 0.72),
      ...hits('rim', [6], 0.64),
      ...every('shaker', 12, 2, 0.36),
    ],
  }),
  rhythm({
    id: 'six-eight-rock',
    label: '6/8 摇滚',
    group: 'other-meter',
    kit: 'acoustic',
    timeSignature: SIX_EIGHT,
    stepsPerBar: 12,
    bpmRange: [70, 155],
    hits: [
      ...hits('kick', [0, 5, 6], 0.9),
      ...hits('snare', [6], 0.92),
      ...every('closed-hat', 12, 2, 0.48),
    ],
  }),
  rhythm({
    id: 'six-eight-fill',
    label: '6/8 轮鼓加花',
    group: 'other-meter',
    kit: 'acoustic',
    timeSignature: SIX_EIGHT,
    stepsPerBar: 12,
    bpmRange: [65, 145],
    hits: [
      ...hits('crash', [0], 0.74),
      ...hits('kick', [0, 5, 6], 0.88),
      ...hits('snare', [6, 11], 0.88),
      ...every('closed-hat', 12, 2, 0.44),
      ...hits('tom-high', [8, 9], 0.66),
      ...hits('tom-low', [10], 0.8),
    ],
  }),
  rhythm({
    id: 'march-two-four',
    label: '2/4 进行曲',
    group: 'other-meter',
    kit: 'acoustic',
    timeSignature: TWO_FOUR,
    stepsPerBar: 8,
    bpmRange: [75, 150],
    hits: [
      ...hits('kick', [0, 4], 0.82),
      ...hits('snare', [2, 6], 0.74),
      ...every('closed-hat', 8, 1, 0.36),
    ],
  }),
  rhythm({
    id: 'country-two-four',
    label: '2/4 Country',
    group: 'other-meter',
    kit: 'acoustic',
    timeSignature: TWO_FOUR,
    stepsPerBar: 8,
    bpmRange: [90, 190],
    hits: [
      ...hits('kick', [0, 4], 0.78),
      ...hits('snare', [2, 6], 0.82),
      ...hits('closed-hat', [0, 2, 4, 6], 0.48),
    ],
  }),
]

export function formatTimeSignature(signature: TimeSignature): string {
  return `${signature[0]}/${signature[1]}`
}

export function rhythmEventPosition(
  measure: TabMeasure,
  rhythmDefinition: PracticeRhythm,
  step: number,
): number {
  if (measure.beats.length === 0) {
    return (
      measure.start +
      (step / rhythmDefinition.stepsPerBar) * measure.duration
    )
  }
  const stepsPerBeat =
    rhythmDefinition.stepsPerBar / measure.beats.length
  const beatIndex = Math.min(
    measure.beats.length - 1,
    Math.floor(step / stepsPerBeat),
  )
  const beat = measure.beats[beatIndex]
  const stepWithinBeat = step - beatIndex * stepsPerBeat
  const swingOffset =
    step % 2 === 1 ? rhythmDefinition.swing * 2 - 1 : 0
  const fraction = Math.max(
    0,
    Math.min(1, (stepWithinBeat + swingOffset) / stepsPerBeat),
  )
  return beat.at + beat.duration * fraction
}

function quantile(sortedValues: number[], fraction: number): number {
  const position = (sortedValues.length - 1) * fraction
  const lower = Math.floor(position)
  const upper = Math.ceil(position)
  const weight = position - lower
  return (
    sortedValues[lower] * (1 - weight) +
    sortedValues[upper] * weight
  )
}

export function practiceTempoProfile(
  measures: TabMeasure[],
  speed = 1,
): PracticeTempoProfile | null {
  const tempos = measures
    .flatMap((measure) => measure.beats)
    .map((beat) => (60 * speed) / beat.duration)
    .filter((tempo) => Number.isFinite(tempo) && tempo >= 20 && tempo <= 320)
    .sort((left, right) => left - right)
  if (tempos.length === 0) return null
  return {
    low: quantile(tempos, 0.1),
    typical: quantile(tempos, 0.5),
    high: quantile(tempos, 0.9),
  }
}

export function countInBeatDurations(
  bpm: number,
  beatsPerBar: number,
): number[] {
  const safeBpm = Number.isFinite(bpm) && bpm > 0 ? bpm : 120
  const safeBeatCount =
    Number.isInteger(beatsPerBar) && beatsPerBar > 0 ? beatsPerBar : 4
  return Array.from({ length: safeBeatCount }, () => 60 / safeBpm)
}

export function rhythmCompatibility(
  rhythmDefinition: PracticeRhythm,
  bpm: number,
  timeSignature: TimeSignature,
): RhythmCompatibility {
  if (
    rhythmDefinition.timeSignature[0] !== timeSignature[0] ||
    rhythmDefinition.timeSignature[1] !== timeSignature[1]
  ) {
    return {
      compatible: false,
      reason: 'time-signature',
      message: `仅适用于 ${formatTimeSignature(rhythmDefinition.timeSignature)}`,
    }
  }
  if (bpm < rhythmDefinition.bpmRange[0]) {
    return {
      compatible: false,
      reason: 'tempo-low',
      message: `至少 ${rhythmDefinition.bpmRange[0]} BPM`,
    }
  }
  if (bpm > rhythmDefinition.bpmRange[1]) {
    return {
      compatible: false,
      reason: 'tempo-high',
      message: `最高 ${rhythmDefinition.bpmRange[1]} BPM`,
    }
  }
  return { compatible: true }
}

export function rhythmCompatibilityForTempoProfile(
  rhythmDefinition: PracticeRhythm,
  profile: PracticeTempoProfile,
  timeSignature: TimeSignature,
): RhythmCompatibility {
  const base = rhythmCompatibility(
    rhythmDefinition,
    profile.typical,
    timeSignature,
  )
  if (base.reason === 'time-signature') return base
  if (profile.low < rhythmDefinition.bpmRange[0]) {
    return {
      compatible: false,
      reason: 'tempo-low',
      message: `歌曲局部低至 ${Math.round(profile.low)} BPM`,
    }
  }
  if (profile.high > rhythmDefinition.bpmRange[1]) {
    return {
      compatible: false,
      reason: 'tempo-high',
      message: `歌曲局部高至 ${Math.round(profile.high)} BPM`,
    }
  }
  return { compatible: true }
}

export function compatibleRhythms(
  bpm: number,
  timeSignature: TimeSignature,
): PracticeRhythm[] {
  return PRACTICE_RHYTHMS.filter(
    (rhythmDefinition) =>
      rhythmCompatibility(rhythmDefinition, bpm, timeSignature).compatible,
  )
}
