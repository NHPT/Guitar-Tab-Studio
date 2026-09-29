import {
  Activity,
  ArrowLeft,
  Check,
  CircleAlert,
  ClipboardCheck,
  Database,
  FlaskConical,
  Gauge,
  Guitar,
  KeyRound,
  LogOut,
  Pause,
  Play,
  RefreshCw,
  Rocket,
  RotateCcw,
  Save,
  ShieldCheck,
  SkipForward,
  UploadCloud,
  Users,
  X,
} from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  advancePromotion,
  apiAssetUrl,
  createDatasetRelease,
  createExperiment,
  createPromotion,
  fetchCommunityAccount,
  fetchCommunityContributions,
  fetchCommunityDashboard,
  fetchCommunityTasks,
  fetchDatasetReleases,
  fetchExperiments,
  fetchInferenceDeployment,
  fetchPromotions,
  fetchTrainingRecipes,
  importCommunityReviews,
  recordPromotionCheck,
  registerCommunityAccount,
  reproduceExperiment,
  retryExperiment,
  submitCommunityTask,
  updateInferenceDeployment,
  withdrawCommunityContribution,
} from './api'
import './CommunityWorkspace.css'
import type {
  CommunityAccount,
  CommunityAnswer,
  CommunityContribution,
  CommunityDashboard,
  CommunityTask,
  DatasetRelease,
  ExperimentRun,
  InferenceDeploymentStatus,
  ModelPromotion,
  TrainingRecipe,
} from './types'

type CommunityView = 'tasks' | 'data' | 'models' | 'operations'
type PromotionCheck = keyof ModelPromotion['checks']

const taskTypeLabels: Record<CommunityTask['type'], string> = {
  'audio-quality': '音频质量',
  'event-presence': '起音确认',
  'missing-event': '漏音检查',
  timing: '时间边界',
  pitch: '音高',
  'string-fret': '弦与品位',
  technique: '演奏技法',
}

const promotionCheckLabels: Record<PromotionCheck, string> = {
  lineage: '谱系完整',
  reproduction: '独立复现',
  'public-validation': '公开验证',
  'sealed-evaluation': '密封评测',
  robustness: '稳健与性能',
  shadow: '影子运行',
  canary: '灰度验证',
}

const phaseIcons = {
  C0: ShieldCheck,
  C1: ClipboardCheck,
  C2: Users,
  C3: Database,
  C4: FlaskConical,
  C5: Rocket,
  C6: Activity,
} as const

function percent(value: number): string {
  return `${Math.round(value * 100)}%`
}

function formatDate(value: string): string {
  return new Date(value).toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function answerLabel(answer: CommunityAnswer): string {
  if (answer.choice === 'yes') return '是'
  if (answer.choice === 'no') return '否'
  if (answer.choice === 'unsure') return '无法判断'
  if (answer.onset !== undefined && answer.offset !== undefined) {
    return `${answer.onset.toFixed(3)}s - ${answer.offset.toFixed(3)}s`
  }
  if (answer.pitch !== undefined) return `MIDI ${answer.pitch}`
  if (answer.string !== undefined) return `${answer.string} 弦 ${answer.fret} 品`
  return answer.technique ?? '未记录'
}

function statusLabel(status: string): string {
  const labels: Record<string, string> = {
    open: '待共识',
    consensus: '已形成共识',
    escalated: '等待裁决',
    withdrawn: '已撤回',
    active: '可用',
    blocked: '已阻断',
    queued: '排队中',
    running: '运行中',
    completed: '已完成',
    failed: '失败',
    cancelled: '已取消',
    candidate: '候选',
    baseline: '基线',
    shadow: '影子',
    canary: '灰度',
    champion: '生产',
    rejected: '已拒绝',
    retired: '已退役',
    ready: '已就绪',
    waiting: '等待输入',
  }
  return labels[status] ?? status
}

function CommunityLogin({
  onRegistered,
}: {
  onRegistered: (token: string, account: CommunityAccount) => void
}) {
  const [alias, setAlias] = useState('')
  const [bootstrapKey, setBootstrapKey] = useState('')
  const [accessKey, setAccessKey] = useState('')
  const [mode, setMode] = useState<'create' | 'access'>('create')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  const submit = async () => {
    setSubmitting(true)
    setError('')
    try {
      if (mode === 'access') {
        const normalizedKey = accessKey.trim()
        const currentAccount = await fetchCommunityAccount(normalizedKey)
        window.localStorage.setItem('gts-community-token', normalizedKey)
        onRegistered(normalizedKey, currentAccount)
        return
      }
      const result = await registerCommunityAccount(
        alias,
        bootstrapKey || undefined,
      )
      window.localStorage.setItem('gts-community-token', result.token)
      onRegistered(result.token, result.account)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '账号创建失败')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <main className="community-login">
      <a className="community-back" href="/" title="返回工作台">
        <ArrowLeft aria-hidden="true" />
        <span className="sr-only">返回工作台</span>
      </a>
      <section aria-labelledby="community-login-title">
        <span className="community-brand-mark">
          <Guitar aria-hidden="true" strokeWidth={2.5} />
        </span>
        <p>HackAll Tab</p>
        <h1 id="community-login-title">
          {mode === 'create' ? '创建贡献者身份' : '使用访问密钥'}
        </h1>
        <div className="community-auth-mode">
          <button
            type="button"
            className={mode === 'create' ? 'is-active' : ''}
            onClick={() => setMode('create')}
          >
            创建身份
          </button>
          <button
            type="button"
            className={mode === 'access' ? 'is-active' : ''}
            onClick={() => setMode('access')}
          >
            已有密钥
          </button>
        </div>
        <div className="community-form-stack">
          {mode === 'create' ? (
            <>
              <label>
                <span>贡献者代号</span>
                <input
                  value={alias}
                  maxLength={32}
                  autoComplete="username"
                  placeholder="例如 string-player"
                  onChange={(event) => setAlias(event.target.value)}
                />
              </label>
              <label>
                <span>引导密钥</span>
                <input
                  value={bootstrapKey}
                  type="password"
                  autoComplete="off"
                  placeholder="仅首次部署需要"
                  onChange={(event) => setBootstrapKey(event.target.value)}
                />
              </label>
            </>
          ) : (
            <label>
              <span>访问密钥</span>
              <input
                value={accessKey}
                type="password"
                autoComplete="off"
                placeholder="粘贴账号访问密钥"
                onChange={(event) => setAccessKey(event.target.value)}
              />
            </label>
          )}
        </div>
        {error && <div className="community-inline-error">{error}</div>}
        <button
          className="community-primary"
          type="button"
          disabled={
            submitting ||
            (mode === 'create'
              ? alias.trim().length < 2
              : accessKey.trim().length < 20)
          }
          onClick={() => void submit()}
        >
          {mode === 'create' ? (
            <Users aria-hidden="true" />
          ) : (
            <KeyRound aria-hidden="true" />
          )}
          {submitting
            ? '正在验证'
            : mode === 'create'
              ? '进入协作台'
              : '使用密钥登录'}
        </button>
      </section>
    </main>
  )
}

export default function CommunityWorkspace() {
  const [token, setToken] = useState(
    () => window.localStorage.getItem('gts-community-token') ?? '',
  )
  const [account, setAccount] = useState<CommunityAccount | null>(null)
  const [dashboard, setDashboard] = useState<CommunityDashboard | null>(null)
  const [tasks, setTasks] = useState<CommunityTask[]>([])
  const [contributions, setContributions] = useState<CommunityContribution[]>([])
  const [releases, setReleases] = useState<DatasetRelease[]>([])
  const [recipes, setRecipes] = useState<TrainingRecipe[]>([])
  const [builtinDatasets, setBuiltinDatasets] = useState<
    Array<{ id: string; name: string }>
  >([])
  const [experiments, setExperiments] = useState<ExperimentRun[]>([])
  const [promotions, setPromotions] = useState<ModelPromotion[]>([])
  const [deployment, setDeployment] =
    useState<InferenceDeploymentStatus | null>(null)
  const [deploymentConfig, setDeploymentConfig] = useState<
    InferenceDeploymentStatus['config']
  >({
    shadowSamplePercent: 10,
    canaryTrafficPercent: 5,
    errorBudgetPercent: 5,
    minimumObservations: 10,
  })
  const [view, setView] = useState<CommunityView>('tasks')
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [playing, setPlaying] = useState(false)
  const [releaseName, setReleaseName] = useState('community-residual')
  const [recipeId, setRecipeId] = useState('dataset-audit-v1')
  const [datasetId, setDatasetId] = useState('builtin:guitarset-v1')
  const [selectedExperimentId, setSelectedExperimentId] = useState('')
  const [selectedPromotionId, setSelectedPromotionId] = useState('')
  const [promotionCheck, setPromotionCheck] =
    useState<PromotionCheck>('shadow')
  const [promotionSummary, setPromotionSummary] = useState('')
  const audioRef = useRef<HTMLAudioElement>(null)
  const taskStartedAt = useRef(0)

  const isOwner = Boolean(account?.roles.includes('owner'))
  const canMaintainModels = Boolean(
    isOwner || account?.roles.includes('model-maintainer'),
  )
  const activeTask = tasks[0]

  const refreshPublic = useCallback(async () => {
    setDashboard(await fetchCommunityDashboard())
  }, [])

  const refreshPrivate = useCallback(
    async (credential: string, knownAccount?: CommunityAccount) => {
      const currentAccount =
        knownAccount ?? (await fetchCommunityAccount(credential))
      const owner = currentAccount.roles.includes('owner')
      const modelMaintainer =
        owner || currentAccount.roles.includes('model-maintainer')
      const [
        nextTasks,
        nextContributions,
        nextReleases,
        recipeData,
        nextExperiments,
        nextPromotions,
        nextDeployment,
      ] = await Promise.all([
        fetchCommunityTasks(credential),
        fetchCommunityContributions(credential),
        fetchDatasetReleases(),
        fetchTrainingRecipes(),
        fetchExperiments(credential),
        modelMaintainer ? fetchPromotions(credential) : Promise.resolve([]),
        modelMaintainer
          ? fetchInferenceDeployment(credential)
          : Promise.resolve(null),
      ])
      setAccount(currentAccount)
      setTasks(nextTasks)
      setContributions(nextContributions)
      setReleases(nextReleases)
      setRecipes(recipeData.recipes)
      setBuiltinDatasets(recipeData.builtinDatasets)
      setExperiments(nextExperiments)
      setPromotions(nextPromotions)
      setDeployment(nextDeployment)
      if (nextDeployment) setDeploymentConfig(nextDeployment.config)
      taskStartedAt.current = Date.now()
      if (!datasetId && recipeData.builtinDatasets[0]) {
        setDatasetId(recipeData.builtinDatasets[0].id)
      }
    },
    [datasetId],
  )

  const refreshAll = useCallback(
    async (credential = token, knownAccount?: CommunityAccount) => {
      setLoading(true)
      setError('')
      try {
        await refreshPublic()
        if (credential) await refreshPrivate(credential, knownAccount)
      } catch (cause) {
        const message = cause instanceof Error ? cause.message : '社区服务不可用'
        setError(message)
        if (/凭据/.test(message)) {
          window.localStorage.removeItem('gts-community-token')
          setToken('')
          setAccount(null)
        }
      } finally {
        setLoading(false)
      }
    },
    [refreshPrivate, refreshPublic, token],
  )

  useEffect(() => {
    document.documentElement.classList.add('community-page')
    const refreshTimer = window.setTimeout(() => void refreshAll(), 0)
    return () => {
      window.clearTimeout(refreshTimer)
      document.documentElement.classList.remove('community-page')
    }
  }, [refreshAll])

  const selectedRecipe = recipes.find((recipe) => recipe.id === recipeId)
  const datasetOptions = useMemo(
    () => {
      const compatible = new Set(selectedRecipe?.compatibleDatasets ?? [])
      return [
        ...builtinDatasets,
        ...releases
          .filter((release) => release.status === 'active')
          .map((release) => ({
            id: release.id,
            name: `${release.name} v${release.version}`,
          })),
      ].filter(
        (dataset) => compatible.has('*') || compatible.has(dataset.id),
      )
    },
    [builtinDatasets, releases, selectedRecipe],
  )

  const runAction = async (action: () => Promise<unknown>, message: string) => {
    setBusy(true)
    setError('')
    setNotice('')
    try {
      await action()
      setNotice(message)
      await refreshAll()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '操作失败')
    } finally {
      setBusy(false)
    }
  }

  const submitAnswer = async (answer: CommunityAnswer) => {
    if (!token || !activeTask) return
    const elapsed = Math.max(0, Date.now() - taskStartedAt.current)
    await runAction(async () => {
      audioRef.current?.pause()
      setPlaying(false)
      const result = await submitCommunityTask(
        token,
        activeTask.id,
        answer,
        elapsed,
      )
      setNotice(
        result.calibration
          ? result.calibration.correct
            ? '校准回答正确'
            : '校准回答已记录'
          : result.taskStatus === 'consensus'
            ? '已形成社区共识'
            : '贡献已记录',
      )
    }, '贡献已记录')
  }

  const playTask = async () => {
    if (!activeTask || !audioRef.current) return
    if (playing) {
      audioRef.current.pause()
      setPlaying(false)
      return
    }
    audioRef.current.currentTime = activeTask.context.start
    await audioRef.current.play()
    setPlaying(true)
  }

  const selectedPromotion = promotions.find(
    (promotion) => promotion.id === selectedPromotionId,
  )

  const nextPromotionStage =
    selectedPromotion?.status === 'candidate'
      ? 'shadow'
      : selectedPromotion?.status === 'shadow'
        ? 'canary'
        : selectedPromotion?.status === 'canary'
          ? 'champion'
          : null
  const deploymentOverBudget = Boolean(
    deployment &&
      deployment.observations.total >=
        deployment.config.minimumObservations &&
      deployment.observations.errorRate >
        deployment.config.errorBudgetPercent / 100,
  )

  if (!token || !account) {
    if (loading && token) {
      return (
        <main className="community-loading">
          <RefreshCw aria-hidden="true" />
          <strong>正在连接社区控制面</strong>
        </main>
      )
    }
    return (
      <CommunityLogin
        onRegistered={(credential, nextAccount) => {
          setToken(credential)
          setAccount(nextAccount)
          void refreshAll(credential, nextAccount)
        }}
      />
    )
  }

  return (
    <div className="community-shell">
      <header className="community-header">
        <div className="community-brand">
          <span className="community-brand-mark">
            <Guitar aria-hidden="true" strokeWidth={2.5} />
          </span>
          <div>
            <strong>Guitar Tab Community</strong>
            <span>HackAll · 数据与模型协作</span>
          </div>
        </div>
        <nav aria-label="社区工作区">
          {([
            ['tasks', '标注任务'],
            ['data', '数据与训练'],
            ['models', '模型晋级'],
            ['operations', '运行状态'],
          ] as Array<[CommunityView, string]>).map(([id, label]) => (
            <button
              key={id}
              type="button"
              className={view === id ? 'is-active' : ''}
              onClick={() => setView(id)}
            >
              {label}
            </button>
          ))}
        </nav>
        <div className="community-account">
          <span>
            <strong>{account.alias}</strong>
            <small>
              {account.governanceMode === 'single-maintainer'
                ? '单人维护模式'
                : '团队治理模式'}
            </small>
          </span>
          <a href="/" title="返回工作台">
            <ArrowLeft aria-hidden="true" />
          </a>
          <button
            type="button"
            title="复制访问密钥"
            onClick={() => {
              void navigator.clipboard.writeText(token)
              setNotice('访问密钥已复制')
            }}
          >
            <KeyRound aria-hidden="true" />
          </button>
          <button
            type="button"
            title="退出社区账号"
            onClick={() => {
              window.localStorage.removeItem('gts-community-token')
              setToken('')
              setAccount(null)
            }}
          >
            <LogOut aria-hidden="true" />
          </button>
        </div>
      </header>

      {(error || notice) && (
        <div className={`community-toast ${error ? 'is-error' : ''}`}>
          {error ? <CircleAlert aria-hidden="true" /> : <Check aria-hidden="true" />}
          <span>{error || notice}</span>
          <button
            type="button"
            aria-label="关闭提示"
            onClick={() => {
              setError('')
              setNotice('')
            }}
          >
            <X aria-hidden="true" />
          </button>
        </div>
      )}

      {view === 'tasks' && (
        <main className="community-task-layout">
          <aside className="community-task-summary">
            <span className="community-section-label">贡献队列</span>
            <strong>{tasks.length}</strong>
            <p>个任务等待你的判断</p>
            <dl>
              <div>
                <dt>已提交</dt>
                <dd>{contributions.filter((item) => !item.withdrawnAt).length}</dd>
              </div>
              <div>
                <dt>校准准确率</dt>
                <dd>{percent(dashboard?.quality.calibrationAccuracy ?? 0)}</dd>
              </div>
            </dl>
          </aside>

          <section className="community-task-stage">
            {activeTask ? (
              <>
                <div className="community-task-meta">
                  <span>{taskTypeLabels[activeTask.type]}</span>
                  <span>{activeTask.source.dataset}</span>
                  <span>{activeTask.source.license}</span>
                </div>
                <h1>{activeTask.question}</h1>
                <p>{activeTask.source.title}</p>
                <audio
                  ref={audioRef}
                  src={apiAssetUrl(activeTask.source.audioUrl)}
                  onPause={() => setPlaying(false)}
                  onEnded={() => setPlaying(false)}
                  onTimeUpdate={(event) => {
                    if (event.currentTarget.currentTime >= activeTask.context.end) {
                      event.currentTarget.pause()
                    }
                  }}
                />
                <div className="community-audio">
                  <button
                    type="button"
                    className="community-play"
                    onClick={() => void playTask()}
                    aria-label={playing ? '暂停片段' : '播放片段'}
                  >
                    {playing ? <Pause aria-hidden="true" /> : <Play aria-hidden="true" />}
                  </button>
                  <div>
                    <strong>
                      {activeTask.context.start.toFixed(2)}s -{' '}
                      {activeTask.context.end.toFixed(2)}s
                    </strong>
                    <span>
                      {activeTask.source.subtitle ?? '短片段盲审'}
                    </span>
                  </div>
                  <div className="community-audio-track" aria-hidden="true">
                    <i
                      style={{
                        left: `${
                          activeTask.context.cueAt === undefined
                            ? 50
                            : ((activeTask.context.cueAt -
                                activeTask.context.start) /
                                (activeTask.context.end -
                                  activeTask.context.start)) *
                              100
                        }%`,
                      }}
                    />
                  </div>
                </div>
                <div className="community-answer-row">
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => void submitAnswer({ choice: 'yes' })}
                  >
                    <Check aria-hidden="true" />
                    是
                  </button>
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => void submitAnswer({ choice: 'no' })}
                  >
                    <X aria-hidden="true" />
                    否
                  </button>
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => void submitAnswer({ choice: 'unsure' })}
                  >
                    <SkipForward aria-hidden="true" />
                    无法判断
                  </button>
                </div>
              </>
            ) : (
              <div className="community-empty">
                <ClipboardCheck aria-hidden="true" />
                <h1>当前任务已处理完</h1>
                <button
                  type="button"
                  className="community-secondary"
                  onClick={() => void refreshAll()}
                >
                  <RefreshCw aria-hidden="true" />
                  刷新队列
                </button>
              </div>
            )}
          </section>

          <aside className="community-history">
            <span className="community-section-label">最近贡献</span>
            <div>
              {contributions.slice(0, 8).map((item) => (
                <article key={item.id}>
                  <span>{taskTypeLabels[item.taskType]}</span>
                  <strong>{answerLabel(item.answer)}</strong>
                  <small>{formatDate(item.createdAt)}</small>
                  {!item.withdrawnAt && (
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() =>
                        void runAction(
                          () => withdrawCommunityContribution(token, item.id),
                          '贡献已撤回',
                        )
                      }
                    >
                      <RotateCcw aria-hidden="true" />
                      撤回
                    </button>
                  )}
                </article>
              ))}
              {contributions.length === 0 && <p>暂无贡献记录</p>}
            </div>
          </aside>
        </main>
      )}

      {view === 'data' && (
        <main className="community-operations">
          <section className="community-section">
            <header>
              <div>
                <span className="community-section-label">C1-C3</span>
                <h1>任务来源与数据版本</h1>
              </div>
              {isOwner && (
                <div className="community-actions">
                  <button
                    type="button"
                    className="community-secondary"
                    disabled={busy}
                    onClick={() =>
                      void runAction(
                        () => importCommunityReviews(token, 'residual-onset'),
                        '校准任务已导入',
                      )
                    }
                  >
                    <UploadCloud aria-hidden="true" />
                    导入已审核样本
                  </button>
                  <button
                    type="button"
                    className="community-secondary"
                    disabled={busy}
                    onClick={() =>
                      void runAction(
                        () => importCommunityReviews(token, 'residual-onset-v2'),
                        'v2 队列已迁移',
                      )
                    }
                  >
                    <UploadCloud aria-hidden="true" />
                    迁移 v2 队列
                  </button>
                </div>
              )}
            </header>
            <div className="community-table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>版本</th>
                    <th>验证级别</th>
                    <th>任务</th>
                    <th>状态</th>
                    <th>清单</th>
                  </tr>
                </thead>
                <tbody>
                  {releases.map((release) => (
                    <tr key={release.id}>
                      <td>{release.name} v{release.version}</td>
                      <td>{release.verification}</td>
                      <td>{release.taskIds.length}</td>
                      <td><i className={`status-${release.status}`} />{statusLabel(release.status)}</td>
                      <td><code>{release.manifestSha256.slice(0, 10)}</code></td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {releases.length === 0 && <div className="community-empty-row">尚未冻结社区数据版本</div>}
            </div>
            {isOwner && (
              <div className="community-inline-form">
                <label>
                  <span>数据版本名称</span>
                  <input
                    value={releaseName}
                    onChange={(event) => setReleaseName(event.target.value)}
                  />
                </label>
                <button
                  type="button"
                  className="community-primary"
                  disabled={busy || !releaseName.trim()}
                  onClick={() =>
                    void runAction(
                      () => createDatasetRelease(token, releaseName),
                      '数据版本已冻结',
                    )
                  }
                >
                  <Database aria-hidden="true" />
                  冻结共识数据
                </button>
              </div>
            )}
          </section>

          <section className="community-section">
            <header>
              <div>
                <span className="community-section-label">C4</span>
                <h2>受控训练</h2>
              </div>
            </header>
            <div className="community-inline-form is-wide">
              <label>
                <span>批准配方</span>
                <select
                  value={recipeId}
                  onChange={(event) => {
                    const nextRecipe = recipes.find(
                      (recipe) => recipe.id === event.target.value,
                    )
                    setRecipeId(event.target.value)
                    const compatible = new Set(
                      nextRecipe?.compatibleDatasets ?? [],
                    )
                    if (
                      !compatible.has('*') &&
                      !compatible.has(datasetId)
                    ) {
                      const nextDataset = [
                        ...builtinDatasets,
                        ...releases
                          .filter((release) => release.status === 'active')
                          .map((release) => ({ id: release.id })),
                      ].find((dataset) => compatible.has(dataset.id))
                      setDatasetId(nextDataset?.id ?? '')
                    }
                  }}
                >
                  {recipes.map((recipe) => (
                    <option
                      key={recipe.id}
                      value={recipe.id}
                      disabled={recipe.availability !== 'ready'}
                    >
                      {recipe.name}
                      {recipe.availability === 'planned' ? ' · 待部署' : ''}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>冻结数据</span>
                <select
                  value={datasetId}
                  onChange={(event) => setDatasetId(event.target.value)}
                >
                  {datasetOptions.map((dataset) => (
                    <option key={dataset.id} value={dataset.id}>{dataset.name}</option>
                  ))}
                </select>
              </label>
              <button
                type="button"
                className="community-primary"
                disabled={
                  busy ||
                  !selectedRecipe ||
                  selectedRecipe.availability !== 'ready' ||
                  !datasetId
                }
                onClick={() =>
                  void runAction(
                    () =>
                      createExperiment(
                        token,
                        recipeId,
                        datasetId,
                        Object.fromEntries(
                          Object.entries(selectedRecipe?.parameters ?? {}).map(
                            ([key, rule]) => [key, rule.default],
                          ),
                        ),
                      ),
                    '实验已提交',
                  )
                }
              >
                <FlaskConical aria-hidden="true" />
                提交实验
              </button>
            </div>
            <div className="community-table-wrap is-experiments">
              <table>
                <thead>
                  <tr>
                    <th>实验</th>
                    <th>配方</th>
                    <th>数据</th>
                    <th>状态</th>
                    <th>执行器</th>
                    <th>更新时间</th>
                  </tr>
                </thead>
                <tbody>
                  {experiments.map((experiment) => (
                    <tr key={experiment.id}>
                      <td><code>{experiment.id.slice(0, 8)}</code></td>
                      <td>{experiment.recipeId}</td>
                      <td>{experiment.datasetReleaseId}</td>
                      <td>
                        <i className={`status-${experiment.status}`} />
                        {statusLabel(experiment.status)}
                        {experiment.status === 'failed' &&
                          experiment.attemptCount < 3 && (
                            <button
                              type="button"
                              className="community-row-action"
                              title="重新排队"
                              aria-label={`重试实验 ${experiment.id.slice(0, 8)}`}
                              disabled={busy}
                              onClick={() =>
                                void runAction(
                                  () => retryExperiment(token, experiment.id),
                                  '实验已重新排队',
                                )
                              }
                            >
                              <RotateCcw aria-hidden="true" />
                            </button>
                          )}
                        {canMaintainModels &&
                          experiment.status === 'completed' &&
                          experiment.modelVersion && (
                            <button
                              type="button"
                              className="community-row-action"
                              title="创建同谱系复现"
                              aria-label={`复现实验 ${experiment.id.slice(0, 8)}`}
                              disabled={busy}
                              onClick={() =>
                                void runAction(
                                  () =>
                                    reproduceExperiment(token, experiment.id),
                                  '复现实验已排队',
                                )
                              }
                            >
                              <RefreshCw aria-hidden="true" />
                            </button>
                          )}
                      </td>
                      <td>
                        <span className="community-runner-state">
                          <strong>
                            {experiment.lease?.runnerId ??
                              experiment.lastRunnerId ??
                              (experiment.status === 'queued'
                                ? '等待领取'
                                : '控制台')}
                          </strong>
                          {experiment.attemptCount > 0 && (
                            <small>
                              第 {experiment.attemptCount} 次
                              {experiment.lease
                                ? ` · 至 ${formatDate(experiment.lease.expiresAt)}`
                                : experiment.executionEvidence
                                  ? ` · ${experiment.executionEvidence.isolation}`
                                  : ''}
                            </small>
                          )}
                        </span>
                      </td>
                      <td>{formatDate(experiment.updatedAt)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {experiments.length === 0 && <div className="community-empty-row">暂无实验运行</div>}
            </div>
          </section>
        </main>
      )}

      {view === 'models' && (
        <main className="community-operations">
          <section className="community-section">
            <header>
              <div>
                <span className="community-section-label">C5</span>
                <h1>模型晋级</h1>
              </div>
              {canMaintainModels && (
                <div className="community-actions">
                  <select
                    aria-label="已完成模型实验"
                    value={selectedExperimentId}
                    onChange={(event) => setSelectedExperimentId(event.target.value)}
                  >
                    <option value="">选择已完成实验</option>
                    {experiments
                      .filter(
                        (experiment) =>
                          experiment.status === 'completed' &&
                          experiment.modelVersion,
                      )
                      .map((experiment) => (
                        <option key={experiment.id} value={experiment.id}>
                          {experiment.modelVersion}
                        </option>
                      ))}
                  </select>
                  <button
                    type="button"
                    className="community-primary"
                    disabled={busy || !selectedExperimentId}
                    onClick={() =>
                      void runAction(
                        () => createPromotion(token, selectedExperimentId),
                        '模型已进入候选阶段',
                      )
                    }
                  >
                    <Rocket aria-hidden="true" />
                    提名候选
                  </button>
                </div>
              )}
            </header>
            <div className="community-model-list">
              {promotions.map((promotion) => (
                <button
                  type="button"
                  key={promotion.id}
                  className={selectedPromotionId === promotion.id ? 'is-active' : ''}
                  onClick={() => setSelectedPromotionId(promotion.id)}
                >
                  <span>{promotion.modelVersion}</span>
                  <strong>{statusLabel(promotion.status)}</strong>
                  <small>
                    {Object.values(promotion.checks).filter((check) => check?.passed).length}
                    {' / 7 门禁'}
                  </small>
                </button>
              ))}
              {promotions.length === 0 && <div className="community-empty-row">暂无模型候选</div>}
            </div>
            {canMaintainModels && deployment && (
              <div className="community-deployment-panel">
                <div className="community-deployment-heading">
                  <div>
                    <Gauge aria-hidden="true" />
                    <span>
                      <strong>推理部署</strong>
                      <small>
                        {deployment.active?.modelVersion ?? '内置基线'}
                      </small>
                    </span>
                  </div>
                  <i
                    className={
                      deploymentOverBudget ? 'is-warning' : 'is-healthy'
                    }
                  >
                    {deploymentOverBudget
                      ? '超出错误预算'
                      : statusLabel(deployment.active?.mode ?? 'baseline')}
                  </i>
                </div>
                <dl className="community-deployment-metrics">
                  <div>
                    <dt>当前模式</dt>
                    <dd>{statusLabel(deployment.active?.mode ?? 'baseline')}</dd>
                  </div>
                  <div>
                    <dt>有效观测</dt>
                    <dd>
                      {deployment.observations.total}
                      <small> / {deployment.config.minimumObservations}</small>
                    </dd>
                  </div>
                  <div>
                    <dt>错误率</dt>
                    <dd>{percent(deployment.observations.errorRate)}</dd>
                  </div>
                  <div>
                    <dt>自动回退</dt>
                    <dd>{deployment.observations.fallbackCount}</dd>
                  </div>
                  <div>
                    <dt>平均耗时</dt>
                    <dd>
                      {deployment.observations.averageDurationMs > 0
                        ? `${deployment.observations.averageDurationMs} ms`
                        : '—'}
                    </dd>
                  </div>
                </dl>
                <form
                  className="community-deployment-form"
                  onSubmit={(event) => {
                    event.preventDefault()
                    void runAction(
                      () =>
                        updateInferenceDeployment(token, deploymentConfig),
                      '推理部署策略已更新',
                    )
                  }}
                >
                  <label>
                    <span>影子采样</span>
                    <input
                      aria-label="影子采样比例"
                      type="number"
                      min="0"
                      max="100"
                      step="1"
                      value={deploymentConfig.shadowSamplePercent}
                      onChange={(event) =>
                        setDeploymentConfig((current) => ({
                          ...current,
                          shadowSamplePercent: Number(event.target.value),
                        }))
                      }
                    />
                    <small>%</small>
                  </label>
                  <label>
                    <span>灰度流量</span>
                    <input
                      aria-label="灰度流量比例"
                      type="number"
                      min="0"
                      max="50"
                      step="1"
                      value={deploymentConfig.canaryTrafficPercent}
                      onChange={(event) =>
                        setDeploymentConfig((current) => ({
                          ...current,
                          canaryTrafficPercent: Number(event.target.value),
                        }))
                      }
                    />
                    <small>%</small>
                  </label>
                  <label>
                    <span>错误预算</span>
                    <input
                      aria-label="错误预算比例"
                      type="number"
                      min="0"
                      max="50"
                      step="0.1"
                      value={deploymentConfig.errorBudgetPercent}
                      onChange={(event) =>
                        setDeploymentConfig((current) => ({
                          ...current,
                          errorBudgetPercent: Number(event.target.value),
                        }))
                      }
                    />
                    <small>%</small>
                  </label>
                  <label>
                    <span>最小观测</span>
                    <input
                      aria-label="最小观测数量"
                      type="number"
                      min="1"
                      max="100"
                      step="1"
                      value={deploymentConfig.minimumObservations}
                      onChange={(event) =>
                        setDeploymentConfig((current) => ({
                          ...current,
                          minimumObservations: Number(event.target.value),
                        }))
                      }
                    />
                  </label>
                  <button
                    type="submit"
                    className="community-secondary"
                    disabled={busy}
                  >
                    <Save aria-hidden="true" />
                    保存策略
                  </button>
                </form>
                {deployment.recent.length > 0 && (
                  <div className="community-deployment-history">
                    <strong>近期推理记录</strong>
                    {deployment.recent.slice(0, 5).map((observation) => (
                      <div key={observation.id}>
                        <i
                          className={
                            observation.success && !observation.fallbackUsed
                              ? 'is-success'
                              : 'is-warning'
                          }
                        />
                        <span>{statusLabel(observation.mode)}</span>
                        <code>{observation.jobId.slice(0, 8)}</code>
                        <small>
                          {observation.fallbackUsed
                            ? '已回退'
                            : observation.success
                              ? '成功'
                              : '失败'}
                        </small>
                        <time dateTime={observation.recordedAt}>
                          {formatDate(observation.recordedAt)}
                        </time>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
            {canMaintainModels && selectedPromotion && (
              <div className="community-gate-panel">
                <div className="community-gate-grid">
                  {(Object.keys(promotionCheckLabels) as PromotionCheck[]).map(
                    (check) => (
                      <span
                        key={check}
                        className={
                          selectedPromotion.checks[check]?.passed
                            ? 'is-passed'
                            : ''
                        }
                      >
                        {selectedPromotion.checks[check]?.passed && (
                          <Check aria-hidden="true" />
                        )}
                        {promotionCheckLabels[check]}
                      </span>
                    ),
                  )}
                </div>
                <div className="community-inline-form is-wide">
                  <select
                    aria-label="门禁检查项"
                    value={promotionCheck}
                    onChange={(event) =>
                      setPromotionCheck(event.target.value as PromotionCheck)
                    }
                  >
                    {(['shadow', 'canary'] as PromotionCheck[]).map((check) => (
                        <option key={check} value={check}>
                          {promotionCheckLabels[check]}
                        </option>
                      ))}
                  </select>
                  <input
                    aria-label="门禁摘要"
                    value={promotionSummary}
                    placeholder="填写证据摘要"
                    onChange={(event) => setPromotionSummary(event.target.value)}
                  />
                  <button
                    type="button"
                    className="community-secondary"
                    disabled={busy || !promotionSummary.trim()}
                    onClick={() =>
                      void runAction(
                        () =>
                          recordPromotionCheck(token, selectedPromotion.id, {
                            check: promotionCheck,
                            passed: true,
                            summary: promotionSummary,
                          }),
                        '门禁结果已记录',
                      )
                    }
                  >
                    <ShieldCheck aria-hidden="true" />
                    记录通过
                  </button>
                  {nextPromotionStage && (
                    <button
                      type="button"
                      className="community-primary"
                      disabled={busy}
                      onClick={() =>
                        void runAction(
                          () =>
                            advancePromotion(
                              token,
                              selectedPromotion.id,
                              nextPromotionStage,
                            ),
                          `模型已进入${statusLabel(nextPromotionStage)}`,
                        )
                      }
                    >
                      <Rocket aria-hidden="true" />
                      晋级至{statusLabel(nextPromotionStage)}
                    </button>
                  )}
                </div>
              </div>
            )}
          </section>
        </main>
      )}

      {view === 'operations' && dashboard && (
        <main className="community-operations">
          <section className="community-metrics" aria-label="社区运行指标">
            {[
              ['贡献者', dashboard.counts.contributors],
              ['待处理任务', dashboard.counts.openTasks],
              ['已形成共识', dashboard.counts.consensusTasks],
              ['活动数据版本', dashboard.counts.activeReleases],
              ['训练队列', dashboard.counts.queuedExperiments],
              ['模型候选', dashboard.counts.promotionCandidates],
            ].map(([label, value]) => (
              <div key={label}>
                <span>{label}</span>
                <strong>{value}</strong>
              </div>
            ))}
          </section>
          <section className="community-section">
            <header>
              <div>
                <span className="community-section-label">C0-C6</span>
                <h1>阶段状态</h1>
              </div>
            </header>
            <div className="community-phase-list">
              {dashboard.phases.map((phase) => {
                const Icon = phaseIcons[phase.id]
                return (
                  <article key={phase.id}>
                    <Icon aria-hidden="true" />
                    <span>{phase.id}</span>
                    <div>
                      <strong>{phase.label}</strong>
                      <small>{phase.detail}</small>
                    </div>
                    <i className={`phase-${phase.status}`}>
                      {statusLabel(phase.status)}
                    </i>
                  </article>
                )
              })}
            </div>
          </section>
          <section className="community-section">
            <header>
              <div>
                <span className="community-section-label">C6</span>
                <h2>质量与维护</h2>
              </div>
            </header>
            <dl className="community-quality">
              <div><dt>共识覆盖</dt><dd>{percent(dashboard.quality.agreementRate)}</dd></div>
              <div><dt>校准准确率</dt><dd>{percent(dashboard.quality.calibrationAccuracy)}</dd></div>
              <div><dt>权利完整率</dt><dd>{percent(dashboard.quality.rightsCoverage)}</dd></div>
            </dl>
            <div className="community-maintenance">
              {dashboard.maintenance.map((item) => (
                <p key={item.message} className={`is-${item.severity}`}>
                  {item.severity === 'warning' ? (
                    <CircleAlert aria-hidden="true" />
                  ) : (
                    <Activity aria-hidden="true" />
                  )}
                  {item.message}
                </p>
              ))}
            </div>
          </section>
        </main>
      )}
    </div>
  )
}
