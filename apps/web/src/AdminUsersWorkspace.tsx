import {
  Check,
  CircleAlert,
  LoaderCircle,
  Plus,
  UserRoundCog,
} from 'lucide-react'
import { type FormEvent, useEffect, useState } from 'react'
import {
  createAdminUser,
  fetchAdminAudit,
  fetchAdminUsers,
  updateAdminUser,
} from './api'
import type {
  AdminAuditEvent,
  AdminRole,
  AdminUser,
} from './types'
import './AdminControlPages.css'

const roleLabels: Record<AdminRole, string> = {
  owner: '所有者',
  reviewer: '复核员',
  adjudicator: '裁决员',
  'model-maintainer': '模型维护员',
  viewer: '只读访客',
}

export default function AdminUsersWorkspace() {
  const [users, setUsers] = useState<AdminUser[]>([])
  const [audit, setAudit] = useState<AdminAuditEvent[]>([])
  const [username, setUsername] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [password, setPassword] = useState('')
  const [role, setRole] = useState<AdminRole>('reviewer')
  const [reviewerAlias, setReviewerAlias] = useState('reviewer-b')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  const refresh = async () => {
    const [nextUsers, nextAudit] = await Promise.all([
      fetchAdminUsers(),
      fetchAdminAudit(),
    ])
    setUsers(nextUsers)
    setAudit(nextAudit)
  }

  useEffect(() => {
    let active = true
    Promise.all([fetchAdminUsers(), fetchAdminAudit()])
      .then(([nextUsers, nextAudit]) => {
        if (!active) return
        setUsers(nextUsers)
        setAudit(nextAudit)
      })
      .catch((reason: unknown) => {
        if (active) {
          setError(reason instanceof Error ? reason.message : '账号数据加载失败')
        }
      })
    return () => {
      active = false
    }
  }, [])

  const create = async (event: FormEvent) => {
    event.preventDefault()
    if (busy) return
    setBusy(true)
    setError('')
    setNotice('')
    try {
      await createAdminUser({
        username,
        password,
        displayName,
        role,
        reviewerAlias: role === 'reviewer' ? reviewerAlias : undefined,
      })
      setUsername('')
      setDisplayName('')
      setPassword('')
      setNotice('账号已创建')
      await refresh()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '账号创建失败')
    } finally {
      setBusy(false)
    }
  }

  const toggleUser = async (user: AdminUser) => {
    setBusy(true)
    setError('')
    try {
      await updateAdminUser(user.id, { active: !user.active })
      await refresh()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '账号状态更新失败')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="admin-control-page">
      <header className="admin-page-heading">
        <div>
          <span>ACCESS CONTROL</span>
          <h1>用户与审计</h1>
        </div>
      </header>

      <div className="admin-control-columns">
        <section className="admin-control-primary">
          <header>
            <div>
              <span>账号</span>
              <strong>{users.length}</strong>
            </div>
          </header>
          <div className="admin-user-list">
            {users.map((user) => (
              <article key={user.id}>
                <UserRoundCog aria-hidden="true" />
                <div>
                  <strong>{user.displayName}</strong>
                  <span>
                    {user.username} · {roleLabels[user.role]}
                  </span>
                </div>
                <code>{user.reviewerAlias ?? '未绑定队列'}</code>
                <button
                  type="button"
                  disabled={busy}
                  className={user.active ? 'is-active' : ''}
                  onClick={() => void toggleUser(user)}
                >
                  {user.active ? '已启用' : '已停用'}
                </button>
              </article>
            ))}
          </div>
        </section>

        <aside className="admin-control-inspector">
          <form onSubmit={(event) => void create(event)}>
            <header>
              <Plus aria-hidden="true" />
              <strong>创建账号</strong>
            </header>
            <label>
              <span>账号</span>
              <input
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                required
              />
            </label>
            <label>
              <span>显示名称</span>
              <input
                value={displayName}
                onChange={(event) => setDisplayName(event.target.value)}
              />
            </label>
            <label>
              <span>初始密码</span>
              <input
                type="password"
                minLength={12}
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                required
              />
            </label>
            <label>
              <span>角色</span>
              <select
                value={role}
                onChange={(event) =>
                  setRole(event.target.value as AdminRole)}
              >
                {Object.entries(roleLabels).map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
            {role === 'reviewer' ? (
              <label>
                <span>审核者标识</span>
                <select
                  value={reviewerAlias}
                  onChange={(event) => setReviewerAlias(event.target.value)}
                >
                  <option value="reviewer-a">reviewer-a</option>
                  <option value="reviewer-b">reviewer-b</option>
                </select>
              </label>
            ) : null}
            {error ? (
              <p className="admin-control-message is-error" role="alert">
                <CircleAlert aria-hidden="true" />
                {error}
              </p>
            ) : null}
            {notice ? (
              <p className="admin-control-message is-success">
                <Check aria-hidden="true" />
                {notice}
              </p>
            ) : null}
            <button className="admin-control-submit" type="submit" disabled={busy}>
              {busy ? <LoaderCircle className="spin" /> : <Plus />}
              创建账号
            </button>
          </form>
        </aside>
      </div>

      <section className="admin-audit-section">
        <header>
          <span>最近审计</span>
          <small>{audit.length} 条</small>
        </header>
        <div>
          {audit.slice(0, 40).map((event, index) => (
            <article key={`${event.at}-${index}`}>
              <time>{new Date(event.at).toLocaleString('zh-CN', { hour12: false })}</time>
              <strong>{event.action}</strong>
              <span>{event.actorUsername ?? 'system'}</span>
            </article>
          ))}
        </div>
      </section>
    </div>
  )
}
