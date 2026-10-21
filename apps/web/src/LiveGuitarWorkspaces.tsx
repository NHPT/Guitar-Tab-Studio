import {
  midiFrequency,
  tunerReadingForTarget,
} from '@gts/core'
import {
  Activity,
  ArrowLeft,
  Gauge,
  Guitar,
  Mic,
  Play,
  Radio,
  Square,
  Trash2,
  Volume2,
} from 'lucide-react'
import { useState, type CSSProperties } from 'react'
import type { LiveGuitarController } from './useLiveGuitar'

interface WorkspaceProps {
  controller: LiveGuitarController
  onBack: () => void
  t: (source: string) => string
}

const standardTuning = [
  { midi: 40, note: 'E2', string: 6 },
  { midi: 45, note: 'A2', string: 5 },
  { midi: 50, note: 'D3', string: 4 },
  { midi: 55, note: 'G3', string: 3 },
  { midi: 59, note: 'B3', string: 2 },
  { midi: 64, note: 'E4', string: 1 },
] as const

function InputToggle({
  active,
  activeLabel,
  idleLabel,
  onClick,
}: {
  active: boolean
  activeLabel: string
  idleLabel: string
  onClick: () => void
}) {
  return (
    <button
      className={`live-input-toggle${active ? ' is-active' : ''}`}
      type="button"
      aria-label={active ? activeLabel : idleLabel}
      aria-pressed={active}
      onClick={onClick}
    >
      {active ? <Square aria-hidden="true" /> : <Play aria-hidden="true" />}
      <span>{active ? activeLabel : idleLabel}</span>
    </button>
  )
}

export function TunerWorkspace({
  controller,
  onBack,
  t,
}: WorkspaceProps) {
  const [targetMidi, setTargetMidi] = useState<number | null>(null)
  const active =
    controller.status === 'listening' && controller.mode === 'tuner'
  const target = standardTuning.find(
    (candidate) => candidate.midi === targetMidi,
  )
  const reading =
    controller.reading && targetMidi !== null
      ? tunerReadingForTarget(controller.reading, targetMidi)
      : controller.reading
  const cents = reading?.cents ?? 0
  const inTune = Boolean(reading && Math.abs(cents) <= 3)
  const stateLabel = !active
    ? t('调音器已停止')
    : !reading
      ? t('等待音频输入')
      : inTune
        ? t('音准准确')
        : cents < 0
          ? t('偏低')
          : t('偏高')
  const noteLabel = reading
    ? `${reading.note}${reading.octave}`
    : target?.note ?? '--'
  const meterStyle = {
    '--tuner-position': `${50 + Math.max(-50, Math.min(50, cents))}%`,
  } as CSSProperties
  return (
    <section className="live-tool tuner-workspace" aria-labelledby="tuner-title">
      <header>
        <button
          className="practice-back-button"
          type="button"
          aria-label={t('返回分析')}
          title={t('返回分析')}
          onClick={onBack}
        >
          <ArrowLeft aria-hidden="true" />
        </button>
        <div>
          <span className="eyebrow">{t('音频输入')}</span>
          <h2 id="tuner-title">{t('调音器')}</h2>
        </div>
        <InputToggle
          active={active}
          activeLabel={t('停止调音')}
          idleLabel={t('开始调音')}
          onClick={() =>
            active ? controller.stop() : void controller.start('tuner')
          }
        />
      </header>

      <div className={`tuner-stage${inTune ? ' is-in-tune' : ''}`}>
        <div className="tuner-readout">
          <span>
            {target
              ? `${target.string}${t('弦')} · ${target.note}`
              : t('自动识别')}
          </span>
          <div className="tuner-note">
            <strong className={!reading && !target ? 'is-empty' : ''}>
              {noteLabel}
            </strong>
            <small>
              {reading ? `${reading.frequency.toFixed(2)} Hz` : '-- Hz'}
            </small>
          </div>
          <b>{stateLabel}</b>
        </div>

        <div className="tuner-meter">
          <div className="tuner-meter-labels" aria-hidden="true">
            <span>{t('偏低')}</span>
            <strong>
              {reading
                ? `${cents >= 0 ? '+' : ''}${cents.toFixed(1)} cent`
                : '0.0 cent'}
            </strong>
            <span>{t('偏高')}</span>
          </div>
          <div
            className="tuner-scale"
            aria-label={t('音分偏差')}
            aria-valuemin={-50}
            aria-valuemax={50}
            aria-valuenow={Number(cents.toFixed(1))}
            role="meter"
            style={meterStyle}
          >
            <div aria-hidden="true">
              {Array.from({ length: 21 }, (_, index) => {
                const value = index * 5 - 50
                return (
                  <i
                    className={
                      value === 0
                        ? 'is-center'
                        : value % 25 === 0
                          ? 'is-major'
                          : ''
                    }
                    key={value}
                  />
                )
              })}
            </div>
            <b aria-hidden="true" />
          </div>
          <div className="tuner-scale-values" aria-hidden="true">
            <span>-50</span>
            <span>-25</span>
            <span>0</span>
            <span>+25</span>
            <span>+50</span>
          </div>
        </div>

        <div className="tuner-telemetry">
          <span>
            <Gauge aria-hidden="true" />
            A4 440 Hz
          </span>
          <span>
            <Activity aria-hidden="true" />
            {reading ? `${Math.round(reading.confidence * 100)}%` : '--'}
          </span>
          <div className="input-level">
            <Mic aria-hidden="true" />
            <i>
              <b
                style={{
                  width: `${Math.min(100, controller.level * 850)}%`,
                }}
              />
            </i>
          </div>
        </div>

        <div className="tuner-targets">
          <div>
            <strong>{t('目标弦')}</strong>
            <small>{t('标准调弦')}</small>
          </div>
          <div className="tuner-strings" aria-label={t('标准调弦')}>
            <button
              className={targetMidi === null ? 'is-active' : ''}
              type="button"
              aria-pressed={targetMidi === null}
              onClick={() => setTargetMidi(null)}
            >
              <span>AUTO</span>
              <small>{t('自动')}</small>
            </button>
            {standardTuning.map((candidate) => {
              const detected =
                reading &&
                reading.midi === candidate.midi &&
                targetMidi === null
              const selected = targetMidi === candidate.midi
              return (
                <button
                  className={selected || detected ? 'is-active' : ''}
                  type="button"
                  aria-label={`${candidate.string}${t('弦')} ${candidate.note}`}
                  aria-pressed={selected}
                  key={candidate.midi}
                  onClick={() => setTargetMidi(candidate.midi)}
                >
                  <span>{candidate.note}</span>
                  <small>{midiFrequency(candidate.midi).toFixed(1)}</small>
                </button>
              )
            })}
          </div>
        </div>
      </div>
      {controller.error ? (
        <p className="live-input-error" role="alert">
          {controller.error}
        </p>
      ) : null}
    </section>
  )
}

export function LiveTranscriptionWorkspace({
  controller,
  onBack,
  t,
}: WorkspaceProps) {
  const active =
    controller.status === 'listening' &&
    controller.mode === 'transcription'
  const notes = controller.notes.slice(-16)
  const current = notes.at(-1)
  return (
    <section
      className="live-tool live-transcription-workspace"
      aria-labelledby="live-transcription-title"
    >
      <header>
        <button
          className="practice-back-button"
          type="button"
          aria-label={t('返回分析')}
          title={t('返回分析')}
          onClick={onBack}
        >
          <ArrowLeft aria-hidden="true" />
        </button>
        <div>
          <span className="eyebrow">{t('音频输入')}</span>
          <h2 id="live-transcription-title">{t('实时记谱')}</h2>
        </div>
        <div className="live-tool-actions">
          <button
            className="icon-button"
            type="button"
            aria-label={t('清空实时曲谱')}
            title={t('清空实时曲谱')}
            disabled={controller.notes.length === 0}
            onClick={controller.clear}
          >
            <Trash2 aria-hidden="true" />
          </button>
          <InputToggle
            active={active}
            activeLabel={t('停止实时记谱')}
            idleLabel={t('开始实时记谱')}
            onClick={() =>
              active
                ? controller.stop()
                : void controller.start('transcription')
            }
          />
        </div>
      </header>

      <div className="live-transcription-status">
        <span>
          <Radio aria-hidden="true" />
          {active ? t('正在监听') : t('等待音频输入')}
        </span>
        <strong>{current ? `${current.note}${current.octave}` : '--'}</strong>
        <small>
          {controller.notes.length} {t('个音符')}
        </small>
        <span>
          <Volume2 aria-hidden="true" />
          {Math.round(Math.min(1, controller.level * 10) * 100)}%
        </span>
      </div>

      <div className="live-tab" aria-label={t('实时六线谱')}>
        {([1, 2, 3, 4, 5, 6] as const).map((string) => (
          <div key={string}>
            <span>{string}</span>
            <i>
              {Array.from({ length: 16 }, (_, index) => {
                const note = notes[index]
                return (
                  <b
                    className={note?.string === string ? 'has-note' : ''}
                    key={index}
                  >
                    {note?.string === string ? note.fret : ''}
                  </b>
                )
              })}
            </i>
          </div>
        ))}
      </div>

      <footer className="live-note-strip">
        <Guitar aria-hidden="true" />
        <div>
          {notes.map((note) => (
            <span key={note.id}>
              <strong>{note.note}{note.octave}</strong>
              <small>{note.string}{t('弦')} {note.fret}{t('品')}</small>
            </span>
          ))}
        </div>
      </footer>
      {controller.error ? (
        <p className="live-input-error" role="alert">
          {controller.error}
        </p>
      ) : null}
    </section>
  )
}
