import {
  Activity,
  Archive,
  AudioLines,
  BookOpenCheck,
  ChevronRight,
  CircleGauge,
  ClipboardCheck,
  Database,
  FlaskConical,
  GitCompareArrows,
  Layers3,
  LockKeyhole,
  LogOut,
  Menu,
  Music2,
  ServerCog,
  ShieldCheck,
  Users,
  X,
} from 'lucide-react'
import { type FormEvent, type ReactNode, useEffect, useState } from 'react'
import {
  bootstrapAdmin,
  fetchAdminDashboard,
  fetchAdminSession,
  loginAdmin,
  logoutAdmin,
} from './api'
import AdjudicationWorkspace from './AdjudicationWorkspace'
import AdminUsersWorkspace from './AdminUsersWorkspace'
import BeatReviewWorkspace from './BeatReviewWorkspace'
import { BrandMark } from './BrandMark'
import EvaluationWorkspace from './EvaluationWorkspace'
import GoldWorkspace from './GoldWorkspace'
import ModelsWorkspace from './ModelsWorkspace'
import ReviewLayerWorkspace from './ReviewLayerWorkspace'
import SystemWorkspace from './SystemWorkspace'
import type {
  AdminDashboard,
  AdminRole,
  AdminSession,
  AdminUser,
} from './types'
import './AdminWorkspace.css'

type AdminIcon = typeof CircleGauge

interface NavigationItem {
  path: string
  label: string
  icon: AdminIcon
  locked?: boolean
}

const navigation: Array<{
  label: string
  items: NavigationItem[]
}> = [
  {
    label: '总览',
    items: [{ path: '/admin', label: '仪表盘', icon: CircleGauge }],
  },
  {
    label: '数据资产',
    items: [
      { path: '/admin/corpus', label: '语料与权利', icon: Database },
      { path: '/admin/gold', label: '金标准版本', icon: Archive },
    ],
  },
  {
    label: '分层复核',
    items: [
      { path: '/admin/reviews/beat', label: '节拍', icon: Activity },
      {
        path: '/admin/reviews/onset',
        label: '起音 / 拨弦',
        icon: AudioLines,
        locked: true,
      },
      {
        path: '/admin/reviews/pitch',
        label: '音高',
        icon: Music2,
        locked: true,
      },
      {
        path: '/admin/reviews/string-fret',
        label: '弦品',
        icon: Layers3,
        locked: true,
      },
      {
        path: '/admin/reviews/chord',
        label: '和弦',
        icon: BookOpenCheck,
        locked: true,
      },
      {
        path: '/admin/reviews/technique',
        label: '技法',
        icon: GitCompareArrows,
        locked: true,
      },
    ],
  },
  {
    label: '质量控制',
    items: [
      {
        path: '/admin/adjudication',
        label: '差异与裁决',
        icon: ClipboardCheck,
      },
      {
        path: '/admin/evaluation',
        label: '评测与门禁',
        icon: ShieldCheck,
      },
      {
        path: '/admin/models',
        label: '模型与实验',
        icon: FlaskConical,
      },
    ],
  },
  {
    label: '管理',
    items: [
      { path: '/admin/users', label: '用户与审计', icon: Users },
      { path: '/admin/system', label: '系统与备份', icon: ServerCog },
    ],
  },
]

const roleLabels: Record<AdminRole, string> = {
  owner: '所有者',
  reviewer: '复核员',
  adjudicator: '裁决员',
  'model-maintainer': '模型维护员',
  viewer: '只读访客',
}

const routeTitles = new Map(
  navigation.flatMap((section) =>
    section.items.map((item) => [item.path, item.label] as const),
  ),
)

function navigate(path: string, replace = false): void {
  if (replace) window.history.replaceState(null, '', path)
  else window.history.pushState(null, '', path)
  window.dispatchEvent(new PopStateEvent('popstate'))
}

function AdminLogin({
  session,
  onAuthenticated,
}: {
  session: AdminSession
  onAuthenticated: (user: AdminUser) => void
}) {
  const setup = !session.configured
  const [username, setUsername] = useState(setup ? 'admin' : '')
  const [displayName, setDisplayName] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    if (busy) return
    setBusy(true)
    setError('')
    try {
      const response = setup
        ? await bootstrapAdmin({ username, password, displayName })
        : await loginAdmin({ username, password })
      onAuthenticated(response.user)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '登录失败')
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="admin-login">
      <section className="admin-login-identity">
        <div className="admin-login-brand">
          <BrandMark />
          <div>
            <strong>Guitar Tab Studio</strong>
            <span>管理控制台</span>
          </div>
        </div>
        <div className="admin-login-signal" aria-hidden="true">
          <AudioLines />
          <i />
          <i />
          <i />
          <i />
          <i />
        </div>
        <p>本地金标准、模型评测与审核状态</p>
      </section>

      <section className="admin-login-panel">
        <form onSubmit={(event) => void submit(event)}>
          <header>
            <span>{setup ? '首次初始化' : '管理员登录'}</span>
            <h1>{setup ? '创建所有者账号' : '进入管理控制台'}</h1>
          </header>
          <label>
            <span>账号</span>
            <input
              autoComplete="username"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              required
            />
          </label>
          {setup ? (
            <label>
              <span>显示名称</span>
              <input
                autoComplete="name"
                placeholder="可选"
                value={displayName}
                onChange={(event) => setDisplayName(event.target.value)}
              />
            </label>
          ) : null}
          <label>
            <span>密码</span>
            <input
              type="password"
              autoComplete={setup ? 'new-password' : 'current-password'}
              minLength={12}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              required
            />
          </label>
          {error ? <p role="alert">{error}</p> : null}
          <button type="submit" disabled={busy}>
            {busy ? '正在处理' : setup ? '初始化并进入' : '登录'}
            <ChevronRight aria-hidden="true" />
          </button>
        </form>
      </section>
    </main>
  )
}

function AdminDashboardPage() {
  const [dashboard, setDashboard] = useState<AdminDashboard | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    fetchAdminDashboard()
      .then((value) => {
        if (active) setDashboard(value)
      })
      .catch((reason: unknown) => {
        if (active) {
          setError(reason instanceof Error ? reason.message : '仪表盘加载失败')
        }
      })
    return () => {
      active = false
    }
  }, [])

  if (error) return <div className="admin-page-error">{error}</div>
  if (!dashboard) return <div className="admin-page-loading">正在加载指标</div>

  const completed =
    dashboard.reviews.submitted + dashboard.reviews.adjudicated
  const progress =
    dashboard.reviews.total > 0
      ? (completed / dashboard.reviews.total) * 100
      : 0

  return (
    <div className="admin-dashboard-page">
      <header className="admin-page-heading">
        <div>
          <span>CONTROL PLANE</span>
          <h1>仪表盘</h1>
        </div>
        <time dateTime={dashboard.generatedAt}>
          {new Date(dashboard.generatedAt).toLocaleString('zh-CN', {
            hour12: false,
          })}
        </time>
      </header>

      <section className="admin-kpi-strip">
        <div>
          <span>待复核</span>
          <strong>{dashboard.reviews.draft}</strong>
          <small>当前队列</small>
        </div>
        <div>
          <span>已提交</span>
          <strong>{dashboard.reviews.submitted}</strong>
          <small>独立审核</small>
        </div>
        <div>
          <span>已裁决</span>
          <strong>{dashboard.reviews.adjudicated}</strong>
          <small>可进入金标准</small>
        </div>
        <div>
          <span>检查完成</span>
          <strong>{dashboard.reviews.checksComplete}</strong>
          <small>4 / 4 项</small>
        </div>
      </section>

      <div className="admin-dashboard-grid">
        <section className="admin-progress-section">
          <header>
            <div>
              <span>节拍复核</span>
              <strong>
                {completed} / {dashboard.reviews.total}
              </strong>
            </div>
            <a
              href="/admin/reviews/beat"
              onClick={(event) => {
                event.preventDefault()
                navigate('/admin/reviews/beat')
              }}
            >
              打开队列
              <ChevronRight aria-hidden="true" />
            </a>
          </header>
          <div className="admin-progress-track">
            <i style={{ width: `${progress}%` }} />
          </div>
          <dl>
            <div>
              <dt>语料版本</dt>
              <dd>{dashboard.corpus?.corpusVersion ?? '未准备'}</dd>
            </div>
            <div>
              <dt>录音 / 片段</dt>
              <dd>
                {dashboard.corpus?.recordings ?? 0} /{' '}
                {dashboard.corpus?.clips ?? 0}
              </dd>
            </div>
            <div>
              <dt>评测窗</dt>
              <dd>{dashboard.corpus?.evaluationDuration ?? 0}s</dd>
            </div>
            <div>
              <dt>审核者</dt>
              <dd>{dashboard.reviewer ?? '未分配'}</dd>
            </div>
          </dl>
        </section>

        <section className="admin-gate-section">
          <header>
            <span>生产门禁</span>
            <strong>尚无裁决金标准</strong>
          </header>
          <dl>
            <div>
              <dt>Beat F1</dt>
              <dd>≥ {dashboard.gates.beatF1.toFixed(2)}</dd>
            </div>
            <div>
              <dt>Downbeat F1</dt>
              <dd>≥ {dashboard.gates.downbeatF1.toFixed(2)}</dd>
            </div>
            <div>
              <dt>中位误差</dt>
              <dd>≤ {dashboard.gates.medianErrorMs}ms</dd>
            </div>
            <div>
              <dt>P95 误差</dt>
              <dd>≤ {dashboard.gates.p95ErrorMs}ms</dd>
            </div>
          </dl>
        </section>
      </div>

      <section className="admin-layer-section">
        <header>
          <span>审核层级</span>
          <small>依赖顺序固定</small>
        </header>
        <div>
          {dashboard.layers.map((layer, index) => (
            <article key={layer.id}>
              <i>{String(index + 1).padStart(2, '0')}</i>
              <strong>{layer.label}</strong>
              <span className={`is-${layer.status}`}>
                {layer.status === 'ready' ? '可复核' : '等待依赖'}
              </span>
            </article>
          ))}
        </div>
      </section>
    </div>
  )
}

function PlaceholderPage({ title, path }: { title: string; path: string }) {
  const reviewLayer = path.startsWith('/admin/reviews/')
  return (
    <div className="admin-placeholder-page">
      <header className="admin-page-heading">
        <div>
          <span>{reviewLayer ? 'REVIEW LAYER' : 'CONTROL PLANE'}</span>
          <h1>{title}</h1>
        </div>
      </header>
      <section>
        <LockKeyhole aria-hidden="true" />
        <strong>{reviewLayer ? '等待上游审核层完成' : '模块尚未启用'}</strong>
        <dl>
          <div>
            <dt>当前状态</dt>
            <dd>锁定</dd>
          </div>
          <div>
            <dt>数据写入</dt>
            <dd>禁止</dd>
          </div>
        </dl>
      </section>
    </div>
  )
}

function AdminShell({
  user,
  path,
  children,
  onLogout,
}: {
  user: AdminUser
  path: string
  children: ReactNode
  onLogout: () => Promise<void>
}) {
  const [menuOpen, setMenuOpen] = useState(false)
  const title = routeTitles.get(path) ?? '管理控制台'

  const follow = (nextPath: string) => {
    setMenuOpen(false)
    navigate(nextPath)
  }

  return (
    <main className="admin-shell">
      <aside className={`admin-sidebar${menuOpen ? ' is-open' : ''}`}>
        <header>
          <BrandMark />
          <div>
            <strong>Guitar Tab Studio</strong>
            <span>管理控制台</span>
          </div>
          <button
            type="button"
            aria-label="关闭菜单"
            title="关闭菜单"
            onClick={() => setMenuOpen(false)}
          >
            <X aria-hidden="true" />
          </button>
        </header>
        <nav>
          {navigation.map((section) => (
            <section key={section.label}>
              <span>{section.label}</span>
              {section.items.map((item) => {
                const Icon = item.icon
                return (
                  <a
                    className={path === item.path ? 'is-current' : ''}
                    href={item.path}
                    key={item.path}
                    onClick={(event) => {
                      event.preventDefault()
                      follow(item.path)
                    }}
                  >
                    <Icon aria-hidden="true" />
                    <strong>{item.label}</strong>
                    {item.locked ? <LockKeyhole aria-hidden="true" /> : null}
                  </a>
                )
              })}
            </section>
          ))}
        </nav>
        <footer>
          <span>{user.displayName}</span>
          <small>{roleLabels[user.role]}</small>
        </footer>
      </aside>
      {menuOpen ? (
        <button
          className="admin-sidebar-scrim"
          type="button"
          aria-label="关闭菜单"
          onClick={() => setMenuOpen(false)}
        />
      ) : null}

      <section className="admin-main">
        <header className="admin-topbar">
          <button
            className="admin-menu-button"
            type="button"
            aria-label="打开菜单"
            title="打开菜单"
            onClick={() => setMenuOpen(true)}
          >
            <Menu aria-hidden="true" />
          </button>
          <div>
            <span>{title}</span>
            <small>本地控制面</small>
          </div>
          <i>
            <span />
            API 正常
          </i>
          <button
            className="admin-logout-button"
            type="button"
            aria-label="退出登录"
            title="退出登录"
            onClick={() => void onLogout()}
          >
            <LogOut aria-hidden="true" />
          </button>
        </header>
        <div className="admin-content">{children}</div>
      </section>
    </main>
  )
}

export default function AdminWorkspace() {
  const [path, setPath] = useState(window.location.pathname)
  const [session, setSession] = useState<AdminSession | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    const storedTheme = window.localStorage.getItem('gts-theme')
    if (storedTheme) {
      document.documentElement.dataset.theme = storedTheme
    }
    document.documentElement.classList.add('admin-page')
    const updatePath = () => setPath(window.location.pathname)
    window.addEventListener('popstate', updatePath)
    return () => {
      document.documentElement.classList.remove('admin-page')
      window.removeEventListener('popstate', updatePath)
    }
  }, [])

  useEffect(() => {
    let active = true
    fetchAdminSession()
      .then((value) => {
        if (!active) return
        setSession(value)
      })
      .catch((reason: unknown) => {
        if (active) {
          setError(reason instanceof Error ? reason.message : '后台连接失败')
        }
      })
    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    if (!session) return
    if (!session.authenticated && path !== '/admin/login') {
      navigate('/admin/login', true)
    } else if (session.authenticated && path === '/admin/login') {
      navigate('/admin', true)
    }
  }, [path, session])

  if (error) return <div className="admin-page-error">{error}</div>
  if (!session) return <div className="admin-page-loading">正在连接管理后台</div>
  if (!session.authenticated || !session.user) {
    return (
      <AdminLogin
        session={session}
        onAuthenticated={(user) => {
          setSession({ configured: true, authenticated: true, user })
          navigate('/admin', true)
        }}
      />
    )
  }

  let content: ReactNode
  if (path === '/admin') {
    content = <AdminDashboardPage />
  } else if (path === '/admin/reviews/beat') {
    content = <BeatReviewWorkspace />
  } else if (path === '/admin/adjudication') {
    content = <AdjudicationWorkspace />
  } else if (path === '/admin/evaluation') {
    content = <EvaluationWorkspace />
  } else if (path === '/admin/gold') {
    content = <GoldWorkspace />
  } else if (path === '/admin/models') {
    content = <ModelsWorkspace />
  } else if (path === '/admin/users') {
    content = <AdminUsersWorkspace />
  } else if (path === '/admin/system') {
    content = <SystemWorkspace />
  } else if (path.startsWith('/admin/reviews/')) {
    const layerId = path.slice('/admin/reviews/'.length) as
      | 'onset'
      | 'pitch'
      | 'string-fret'
      | 'chord'
      | 'technique'
    content = <ReviewLayerWorkspace layerId={layerId} />
  } else {
    content = (
      <PlaceholderPage
        path={path}
        title={routeTitles.get(path) ?? '页面不存在'}
      />
    )
  }

  return (
    <AdminShell
      user={session.user}
      path={path}
      onLogout={async () => {
        await logoutAdmin()
        setSession({ configured: true, authenticated: false, user: null })
        navigate('/admin/login', true)
      }}
    >
      {content}
    </AdminShell>
  )
}
