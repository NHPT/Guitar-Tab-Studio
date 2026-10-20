import {
  ArrowLeft,
  Guitar,
  Mic,
  Play,
  Radio,
  Square,
  Trash2,
  Volume2,
} from 'lucide-react'
import type { LiveGuitarController } from './useLiveGuitar'

interface WorkspaceProps {
  controller: LiveGuitarController
  onBack: () => void
  t: (source: string) => string
}

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
  const active =
    controller.status === 'listening' && controller.mode === 'tuner'
  const reading = controller.reading
  const cents = reading?.cents ?? 0
  const inTune = Boolean(reading && Math.abs(cents) <= 4)
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
        <div className="tuner-scale" aria-label={t('音分偏差')}>
          {[-50, -25, 0, 25, 50].map((value) => (
            <i
              className={value === 0 ? 'is-center' : ''}
              key={value}
            >
              {value > 0 ? `+${value}` : value}
            </i>
          ))}
          <b
            style={{
              left: `${50 + Math.max(-50, Math.min(50, cents))}%`,
            }}
          />
        </div>
        <div className="tuner-note">
          <span>{reading ? `${reading.note}${reading.octave}` : '--'}</span>
          <strong>
            {reading
              ? `${cents >= 0 ? '+' : ''}${cents.toFixed(1)} cent`
              : t('等待音符')}
          </strong>
          <small>
            {reading ? `${reading.frequency.toFixed(1)} Hz` : '440 Hz'}
          </small>
        </div>
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
        <div className="tuner-strings" aria-label={t('标准调弦')}>
          {['E2', 'A2', 'D3', 'G3', 'B3', 'E4'].map((note) => (
            <span
              className={
                reading && `${reading.note}${reading.octave}` === note
                  ? 'is-active'
                  : ''
              }
              key={note}
            >
              {note}
            </span>
          ))}
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
