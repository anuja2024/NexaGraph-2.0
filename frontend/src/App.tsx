
import { Fragment, useEffect, useState, type FormEvent } from 'react'
import {
  Activity, ArrowRight, BookOpen, CheckCircle2, ChevronDown,
  ChevronRight, CircleHelp, FileCheck2, FileText, FolderOpen,
  Layers3, LockKeyhole, LogOut, Menu, MessageSquareText, Search,
  Settings2, ShieldCheck, Sparkles, Users, X, RefreshCw,
  Check, Ban, Clock,
} from 'lucide-react'
import './App.css'

type Project = {
  project_id: string
  name: string
}

type User = {
  user_id: string
  role: string
}

type Citation = Record<string, unknown>

type AnswerResponse = {
  project_id: string
  query: string
  answer: string
  citations: Citation[]
  retrieval_method: string
  answer_status: string
}

type Document = {
  document_id: string
  name: string
  source_file: string
  project_id: string
}

type DocumentsResponse = {
  project_id: string
  documents: Document[]
}

type ApprovalStatus = 'pending' | 'approved' | 'rejected'

type Approval = {
  approval_id: number
  request_id: string
  project_id: string
  employee_query: string
  ai_answer: string
  citations: Citation[]
  retrieval_method: string
  reviewer_decision: ApprovalStatus
  reviewer_comments: string | null
  reviewer_id: string | null
  approval_status: ApprovalStatus
  created_at: string | null
  reviewed_at: string | null
}

type ApprovalResponse = {
  approvals: Approval[]
  total: number
}

type ActivityEvent = {
  event_id: number
  occurred_at: string | null
  user_id: string | null
  event_type: string
  project_id: string | null
  resource_id: string | null
  outcome: string
  request_id: string | null
  details: Record<string, unknown>
}

type ActivityResponse = {
  events: ActivityEvent[]
  count: number
}

type Page = 'Search' | 'Documents' | 'Review queue' | 'Activity' | 'Settings'

const navigation: { label: Page; icon: typeof Search }[] = [
  { label: 'Search', icon: Search },
  { label: 'Documents', icon: FolderOpen },
  { label: 'Review queue', icon: FileCheck2 },
  { label: 'Activity', icon: Activity },
  { label: 'Settings', icon: Settings2 },
]

async function apiRequest<T>(
  path: string,
  token: string,
  options: RequestInit = {},
): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...options,
    headers: {
                  Authorization: `Bearer ${token}`,
      ...(options.body ? { 'Content-Type': 'application/json' } : {}),
      ...options.headers,
    },
  })

  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    const message =
      typeof body.detail === 'string'
        ? body.detail
        : `Request failed (${response.status})`
    throw new Error(message)
  }

  return response.json() as Promise<T>
}

function formatDate(value: string | null) {
  if (!value) return 'Not available'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString()
}

function Brand() {
  return (
    <div className="brand">
      <div className="brand-mark"><Layers3 size={22} strokeWidth={2.3} /></div>
      <div className="brand-copy">
        <span className="brand-name">NexaGraph</span>
        <span className="brand-caption">KNOWLEDGE INTELLIGENCE</span>
      </div>
    </div>
  )
}

function LoginScreen({ onLogin }: {
  onLogin: (identity: string, password: string) => Promise<void>
}) {
  const [identity, setIdentity] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setBusy(true)
    setError('')
    try {
      await onLogin(identity.trim(), password)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Sign-in failed.')
    } finally {
      setBusy(false)
      setPassword('')
    }
  }

  return (
    <main className="login-shell">
      <section className="login-visual">
        <div className="login-visual-grid" />
        <div className="login-visual-content">
          <Brand />
          <div className="visual-message">
            <div className="eyebrow light-eyebrow"><Sparkles size={14} /> YOUR KNOWLEDGE, CONNECTED</div>
            <h1>Make your knowledge<br />work <span>together.</span></h1>
            <p>One intelligent workspace for discovering insights across your organization's documents.</p>
            <div className="visual-graph">
              <div className="graph-line graph-line-one" />
              <div className="graph-line graph-line-two" />
              <div className="graph-line graph-line-three" />
              <div className="graph-node graph-node-main"><Layers3 size={28} /></div>
              <div className="graph-node graph-node-a"><FileText size={19} /></div>
              <div className="graph-node graph-node-b"><Search size={19} /></div>
              <div className="graph-node graph-node-c"><BookOpen size={19} /></div>
              <div className="graph-node graph-node-d"><Sparkles size={19} /></div>
              <span className="graph-label graph-label-a">Documents</span>
              <span className="graph-label graph-label-b">Intelligence</span>
              <span className="graph-label graph-label-c">Knowledge graph</span>
            </div>
          </div>
          <div className="visual-footer"><ShieldCheck size={16} /> Controlled enterprise knowledge access</div>
        </div>
      </section>

      <section className="login-panel">
        <div className="login-mobile-brand"><Brand /></div>
        <div className="login-form-wrap">
          <div className="login-icon"><LockKeyhole size={22} /></div>
          <div className="eyebrow">WORKSPACE ACCESS</div>
          <h2>Welcome back</h2>
          <p className="login-subtitle">Sign in with your assigned employee ID and password.</p>
          <form className="login-form" onSubmit={submit}>
            <label htmlFor="identity">Employee ID</label>
            <input
              id="identity"
              value={identity}
              onChange={(event) => setIdentity(event.target.value)}
              placeholder="Enter your employee ID"
              autoComplete="username"
              required
            />
            <div className="password-label">
              <label htmlFor="password">Password</label>
              <span className="muted-label">Secure sign-in</span>
            </div>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder="Enter your password"
              autoComplete="current-password"
              required
            />
            <button className="primary-button login-submit" type="submit" disabled={busy}>
              {busy ? 'Signing in…' : 'Sign in'} <ArrowRight size={17} />
            </button>
            {error && <div className="inline-notice" role="alert">{error}</div>}
          </form>
          <p className="login-security">
            <ShieldCheck size={16} /> Access is checked by the NexaGraph backend.
          </p>
        </div>
        <div className="login-panel-footer">
          <span>© 2026 NexaGraph</span><span>Enterprise knowledge platform</span>
        </div>
      </section>
    </main>
  )
}

function App() {
  const [token, setToken] = useState('')
  const [user, setUser] = useState<User | null>(null)
  const [projects, setProjects] = useState<Project[]>([])
  const [selectedProject, setSelectedProject] = useState('')
  const [activePage, setActivePage] = useState<Page>('Search')
  const [projectMenuOpen, setProjectMenuOpen] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [answer, setAnswer] = useState<AnswerResponse | null>(null)
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState('')
  const [error, setError] = useState('')

  const [documents, setDocuments] = useState<Document[]>([])
  const [documentsLoading, setDocumentsLoading] = useState(false)
  const [documentsError, setDocumentsError] = useState('')
  const [documentSearch, setDocumentSearch] = useState('')
  const [documentsReload, setDocumentsReload] = useState(0)

  // Review Queue state
  const [approvals, setApprovals] = useState<Approval[]>([])
  const [approvalStatus, setApprovalStatus] = useState<ApprovalStatus>('pending')
  const [approvalsLoading, setApprovalsLoading] = useState(false)
  const [approvalsError, setApprovalsError] = useState('')
  const [approvalsReload, setApprovalsReload] = useState(0)
  const [approvalComments, setApprovalComments] = useState<Record<number, string>>({})
  const [decisionBusy, setDecisionBusy] = useState<number | null>(null)

  // Manager Activity feed state
  const [activityEvents, setActivityEvents] = useState<ActivityEvent[]>([])
  const [activityLoading, setActivityLoading] = useState(false)
  const [activityError, setActivityError] = useState('')
  const [activityReload, setActivityReload] = useState(0)
  const [activitySearch, setActivitySearch] = useState('')
  const [activityOutcome, setActivityOutcome] = useState('all')
  const [activityPage, setActivityPage] = useState(1)
  const [expandedActivityId, setExpandedActivityId] = useState<number | null>(null)
  const activityPageSize = 10

  const project = projects.find((item) => item.project_id === selectedProject)
  const isManager = user?.role === 'manager'

  async function signIn(identity: string, password: string) {
    const response = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: identity, password }),
    })
    const body = await response.json().catch(() => ({}))
    if (!response.ok) {
      throw new Error(body.detail || 'Unable to sign in.')
    }

    const newToken = body.access_token as string
    const profile = await apiRequest<{ user: User }>('/auth/me', newToken)
    const projectData = await apiRequest<{ projects: Project[] }>('/projects', newToken)

    if (!projectData.projects.length) {
      throw new Error('Your account has no project access. Contact your administrator.')
    }

    setToken(newToken)
    setUser(profile.user)
    setProjects(projectData.projects)
    setSelectedProject(projectData.projects[0].project_id)
    setActivePage('Search')
    setAnswer(null)
    setDocuments([])
    setDocumentSearch('')
    setApprovals([])
    setApprovalStatus('pending')
    setApprovalComments({})
    setNotice('')
    setError('')
  }

  function signOut() {
    setToken('')
    setUser(null)
    setProjects([])
    setSelectedProject('')
    setQuery('')
    setAnswer(null)
    setDocuments([])
    setDocumentSearch('')
    setDocumentsError('')
    setApprovals([])
    setApprovalComments({})
    setApprovalsError('')
    setActivityEvents([])
    setActivityError('')
    setNotice('')
    setError('')
    setActivePage('Search')
  }

  useEffect(() => {
    if (!token || activePage !== 'Documents' || !selectedProject) return

    let cancelled = false

    async function loadDocuments() {
      setDocumentsLoading(true)
      setDocumentsError('')
      setDocuments([])

      try {
        const data = await apiRequest<DocumentsResponse>(
          `/projects/${encodeURIComponent(selectedProject)}/documents`,
          token,
        )

        if (!cancelled) {
          setDocuments(Array.isArray(data.documents) ? data.documents : [])
        }
      } catch (err) {
        if (!cancelled) {
          setDocuments([])
          setDocumentsError(
            err instanceof Error ? err.message : 'Could not load documents.',
          )
        }
      } finally {
        if (!cancelled) setDocumentsLoading(false)
      }
    }

    void loadDocuments()

    return () => { cancelled = true }
  }, [token, activePage, selectedProject, documentsReload])

  // Only managers can retrieve the approval queue.
  useEffect(() => {
    if (!token || activePage !== 'Review queue' || !isManager) return

    let cancelled = false

    async function loadApprovals() {
      setApprovalsLoading(true)
      setApprovalsError('')

      try {
        const data = await apiRequest<ApprovalResponse>(
          `/reviews/approvals?status=${approvalStatus}`,
          token,
        )

        if (!cancelled) {
          setApprovals(Array.isArray(data.approvals) ? data.approvals : [])
        }
      } catch (err) {
        if (!cancelled) {
          setApprovals([])
          setApprovalsError(
            err instanceof Error ? err.message : 'Could not load approvals.',
          )
        }
      } finally {
        if (!cancelled) setApprovalsLoading(false)
      }
    }

    void loadApprovals()

    return () => { cancelled = true }
  }, [token, activePage, isManager, approvalStatus, approvalsReload])

  // Only managers can retrieve the audit activity feed.
  useEffect(() => {
    if (!token || activePage !== 'Activity' || !isManager) return

    let cancelled = false

    async function loadActivity() {
      setActivityLoading(true)
      setActivityError('')

      try {
        const data = await apiRequest<ActivityResponse>('/activity?limit=100', token)

        if (!cancelled) {
          setActivityEvents(Array.isArray(data.events) ? data.events : [])
        }
      } catch (err) {
        if (!cancelled) {
          setActivityEvents([])
          setActivityError(
            err instanceof Error ? err.message : 'Could not load activity.',
          )
        }
      } finally {
        if (!cancelled) setActivityLoading(false)
      }
    }

    void loadActivity()

    return () => { cancelled = true }
  }, [token, activePage, isManager, activityReload])

  async function submitDecision(
    approvalId: number,
    decision: 'approved' | 'rejected',
  ) {
    const actionLabel = decision === 'approved' ? 'approve' : 'reject'
    const comments = approvalComments[approvalId]?.trim() || ''

    setDecisionBusy(approvalId)
    setApprovalsError('')
    setNotice('')

    try {
      await apiRequest(
        `/reviews/approvals/${approvalId}/decision`,
        token,
        {
          method: 'POST',
          body: JSON.stringify({
            decision,
            comments: comments || null,
          }),
        },
      )

      setNotice(`Approval #${approvalId} was ${decision} successfully.`)
      setApprovalComments((current) => {
        const next = { ...current }
        delete next[approvalId]
        return next
      })

      // Reload the selected status so the changed record disappears
      // from the pending queue or appears in its historical status.
      setApprovalsReload((current) => current + 1)
    } catch (err) {
      setApprovalsError(
        err instanceof Error
          ? err.message
          : `Unable to ${actionLabel} this answer.`,
      )
    } finally {
      setDecisionBusy(null)
    }
  }

  async function submitSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!query.trim() || !selectedProject) return

    setBusy(true)
    setError('')
    setNotice('')
    setAnswer(null)

    try {
      const result = await apiRequest<AnswerResponse>(
        `/projects/${encodeURIComponent(selectedProject)}/answer`,
        token,
        {
          method: 'POST',
          body: JSON.stringify({ query: query.trim() }),
        },
      )
      setAnswer(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed.')
    } finally {
      setBusy(false)
    }
  }

  const filteredDocuments = documents.filter((document) => {
    const searchValue = documentSearch.trim().toLowerCase()
    if (!searchValue) return true

    return [
      document.name,
      document.source_file,
      document.document_id,
      document.project_id,
    ].some((value) => value?.toLowerCase().includes(searchValue))
  })

  const filteredActivityEvents = activityEvents.filter((event) => {
    const searchValue = activitySearch.trim().toLowerCase()
    const matchesSearch = !searchValue || [
      event.event_type,
      event.user_id || '',
      event.project_id || '',
      event.resource_id || '',
      event.request_id || '',
      event.outcome,
      JSON.stringify(event.details || {}),
    ].some((value) => value.toLowerCase().includes(searchValue))
    const matchesOutcome = activityOutcome === 'all' || event.outcome.toLowerCase() === activityOutcome
    return matchesSearch && matchesOutcome
  })

  const activityPageCount = Math.max(1, Math.ceil(filteredActivityEvents.length / activityPageSize))
  const visibleActivityEvents = filteredActivityEvents.slice(
    (activityPage - 1) * activityPageSize,
    activityPage * activityPageSize,
  )

  if (!token || !user) {
    return <LoginScreen onLogin={signIn} />
  }

  return (
    <div className="app-shell">
      {sidebarOpen && <button aria-label="Close navigation" className="mobile-backdrop" onClick={() => setSidebarOpen(false)} />}
      <aside className={`sidebar ${sidebarOpen ? 'sidebar-open' : ''}`}>
        <div className="sidebar-top">
          <Brand />
          <button className="icon-button mobile-close" aria-label="Close menu" onClick={() => setSidebarOpen(false)}><X size={19} /></button>
        </div>

        <button className="workspace-switcher" onClick={() => setProjectMenuOpen(!projectMenuOpen)}>
          <div className="workspace-avatar"><Layers3 size={18} /></div>
          <div className="workspace-copy"><strong>Knowledge workspace</strong><span>{user.role}</span></div>
          <ChevronDown size={16} className="switcher-chevron" />
        </button>

        {projectMenuOpen && (
          <div className="project-menu">
            <span className="menu-heading">YOUR PROJECTS</span>
            {projects.map((item) => (
              <button
                key={item.project_id}
                className={`project-menu-item ${selectedProject === item.project_id ? 'selected' : ''}`}
                onClick={() => {
                  setSelectedProject(item.project_id)
                  setProjectMenuOpen(false)
                  setAnswer(null)
                  setDocuments([])
                  setDocumentSearch('')
                  setError('')
                } }
              >
                <span className="project-dot blue" />
                <span>{item.name}</span>
                {selectedProject === item.project_id && <CheckCircle2 size={15} />}
              </button>
            ))}
          </div>
        )}

        <div className="sidebar-section-label">WORKSPACE</div>
        <nav className="sidebar-nav" aria-label="Main navigation">
          {navigation.map(({ label, icon: Icon }) => {
            if (!isManager && (label === 'Review queue' || label === 'Activity')) {
              return null
            }

            return (
              <button
                key={label}
                className={`nav-item ${activePage === label ? 'nav-item-active' : ''}`}
                onClick={() => {
                  setActivePage(label)
                  setSidebarOpen(false)
                  setNotice('')
                  setError('')
                  setApprovalsError('')
                } }
              >
                <Icon size={18} strokeWidth={1.8} />
                <span>{label}</span>
                {activePage === label && <span className="active-indicator" />}
              </button>
            )
          })}
        </nav>

      <div className="sidebar-bottom">
        <div className="sidebar-status">
          <div className="status-icon"><ShieldCheck size={17} /></div>
          <div><strong>Access controls</strong><span>Authenticated session</span></div>
          <span className="status-dot" />
        </div>
        <div className="sidebar-user">
          <div className="user-avatar">{user.user_id.slice(0, 1).toUpperCase()}</div>
          <div className="user-info"><strong>{user.user_id}</strong><span>{user.role}</span></div>
          <button className="icon-button" aria-label="Sign out" title="Sign out" onClick={signOut}><LogOut size={17} /></button>
        </div>
      </div>
    </aside><main className="main-area">
        <header className="topbar">
          <div className="topbar-left">
            <button className="icon-button mobile-menu" aria-label="Open menu" onClick={() => setSidebarOpen(true)}><Menu size={21} /></button>
            <div className="breadcrumb"><span>Workspace</span><ChevronRight size={14} /><strong>{activePage}</strong></div>
          </div>
          <div className="topbar-right">
            <span className="preview-pill"><span /> LIVE SESSION</span>
            <button className="icon-button help-button" aria-label="Help" onClick={() => setNotice('Search and project access are provided by the authenticated backend.')}><CircleHelp size={19} /></button>
            <div className="topbar-divider" />
            <button className="topbar-profile" onClick={() => setNotice(`Signed in as ${user.user_id} (${user.role}).`)}>
              <div className="user-avatar small-avatar">{user.user_id.slice(0, 1).toUpperCase()}</div>
              <span>{user.user_id}</span><ChevronDown size={15} />
            </button>
          </div>
        </header>

        <div className="content-area">
          {notice && <div className="notice-banner" role="status"><span>{notice}</span><button onClick={() => setNotice('')} aria-label="Dismiss message"><X size={16} /></button></div>}
          {error && <div className="notice-banner" role="alert"><span>{error}</span><button onClick={() => setError('')} aria-label="Dismiss error"><X size={16} /></button></div>}

          {activePage === 'Search' && (
            <>
              <section className="welcome-row">
                <div>
                  <div className="eyebrow"><Sparkles size={14} /> KNOWLEDGE INTELLIGENCE</div>
                  <h1>Welcome back. <span>What do you need to know?</span></h1>
                  <p>Ask a question about documents in a project you are authorized to access.</p>
                </div>
                <div className="welcome-art"><div className="art-orbit orbit-one" /><div className="art-orbit orbit-two" /><div className="art-core"><Layers3 size={30} /></div><div className="art-dot dot-two" /><div className="art-dot dot-three" /></div>
              </section>

              <section className="search-panel">
                <div className="search-project-label"><span className="project-dot blue" /> SEARCHING IN <strong>{project?.name || 'Your project'}</strong><ChevronDown size={14} /></div>
                <form className="main-search-form" onSubmit={submitSearch}>
                  <Search size={21} />
                  <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Ask a question about your project documents…" aria-label="Search knowledge base" minLength={2} required />
                  <button className="search-submit" type="submit" disabled={busy}>{busy ? 'Searching…' : 'Ask'} <ArrowRight size={16} /></button>
                </form>
                <div className="search-hints"><span><MessageSquareText size={13} /> Grounded answers</span><span><FileText size={13} /> Backend citations</span><span><ShieldCheck size={13} /> Project access controls</span></div>
              </section>

              <div className="section-heading"><div><h2>{answer ? 'Answer' : 'Ready to search'}</h2><p>{answer ? `Project: ${project?.name || answer.project_id}` : 'Your requests are sent to the authenticated NexaGraph API.'}</p></div></div>

              {answer && (
                <section className="results-panel">
                  <div className="results-header"><div><span className="result-label"><Sparkles size={15} /> {answer.retrieval_method}</span><h3>Response</h3></div><span className="sample-badge">{answer.answer_status.replaceAll('_', ' ')}</span></div>
                  <p className="sample-answer">{answer.answer}</p>
                  <div className="source-list">
                    {answer.citations.map((citation, index) => (
                      <div className="source-item" key={`${String(citation.source_file ?? citation.document_id ?? index)}-${index}`}>
                        <div className="source-number">{index + 1}</div>
                        <div className="source-body">
                          <strong>{String(citation.source_file ?? citation.document_id ?? `Source ${index + 1}`)}</strong>
                          <p>{[
                            citation.page_number != null ? `Page ${String(citation.page_number)}` : '',
                            citation.title != null ? String(citation.title) : '',
                          ].filter(Boolean).join(' · ') || 'Citation metadata returned by the backend.'}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                  {answer.citations.length === 0 && <p className="login-subtitle">No citations were returned for this response.</p>}
                </section>
              )}
            </>
          )}

          {activePage === 'Documents' && (
            <section className="page-section">
              <div className="page-title-row">
                <div>
                  <div className="eyebrow">KNOWLEDGE LIBRARY</div>
                  <h1>Documents</h1>
                  <p>Browse documents you are authorized to access{project ? ` in ${project.name}` : ''}.</p>
                </div>
                <button className="secondary-button" type="button" onClick={() => setDocumentsReload((current) => current + 1)} disabled={documentsLoading}>
                  <RefreshCw size={15} /> {documentsLoading ? 'Refreshing…' : 'Refresh'}
                </button>
              </div>

              <div className="toolbar">
                <label className="toolbar-search">
                  <Search size={16} />
                  <input type="search" value={documentSearch} onChange={(event) => setDocumentSearch(event.target.value)} placeholder="Search documents by name or filename…" aria-label="Search documents" />
                </label>
              </div>

              {documentsLoading && <div className="empty-state" role="status">Loading authorized documents…</div>}

              {!documentsLoading && documentsError && (
                <div className="info-card" role="alert">
                  <ShieldCheck size={19} />
                  <div><strong>Could not load documents</strong><p>{documentsError}</p>
                    <button className="text-button" type="button" onClick={() => setDocumentsReload((current) => current + 1)}>Try again <ArrowRight size={14} /></button>
                  </div>
                </div>
              )}

              {!documentsLoading && !documentsError && (
                <>
                  <div className="section-heading documents-heading">
                    <div><h2>{filteredDocuments.length} {filteredDocuments.length === 1 ? 'document' : 'documents'}</h2><p>Registered documents returned by the NexaGraph backend.</p></div>
                  </div>
                  <div className="document-grid document-grid-page">
                    {filteredDocuments.length > 0 ? (
                      filteredDocuments.map((document, index) => {
                        const colors = ['blue', 'green', 'orange']
                        const color = colors[index % colors.length]
                        return (
                          <article className="document-card" key={document.document_id}>
                            <div className={`pdf-icon ${color}`}><FileText /><span>DOC</span></div>
                            <div className="document-details"><strong title={document.name}>{document.name}</strong><span title={document.source_file}>{document.source_file}</span></div>
                            <div className="document-footer"><span className={`category-chip ${color}`}>{document.project_id}</span><span>Registered</span></div>
                          </article>
                        )
                      })
                    ) : (
                      <div className="empty-state">{documents.length === 0 ? 'No authorized documents are registered for this project.' : 'No documents match your search.'}</div>
                    )}
                  </div>
                </>
              )}
            </section>
          )}

          {activePage === 'Review queue' && (
            <section className="page-section">
              <div className="page-title-row">
                <div>
                  <div className="eyebrow">HUMAN-IN-THE-LOOP</div>
                  <h1>Review queue</h1>
                  <p>Review AI-generated answers and their supporting citations.</p>
                </div>
                {isManager && (
                  <button className="secondary-button" type="button" onClick={() => setApprovalsReload((current) => current + 1)} disabled={approvalsLoading}>
                    <RefreshCw size={15} /> {approvalsLoading ? 'Refreshing…' : 'Refresh'}
                  </button>
                )}
              </div>

              {!isManager ? (
                <div className="info-card">
                  <ShieldCheck size={19} />
                  <div>
                    <strong>Manager access required</strong>
                    <p>The approval queue is restricted to authenticated users with the manager role.</p>
                  </div>
                </div>
              ) : (
                <>
                  <div className="review-status-tabs" role="tablist" aria-label="Approval status">
                    {(['pending', 'approved', 'rejected'] as ApprovalStatus[]).map((status) => (
                      <button
                        key={status}
                        type="button"
                        role="tab"
                        aria-selected={approvalStatus === status}
                        className={`review-status-tab ${approvalStatus === status ? 'active' : ''}`}
                        onClick={() => {
                          setApprovalStatus(status)
                          setApprovalsError('')
                        } }
                      >
                        {status === 'pending' ? <Clock size={15} /> : status === 'approved' ? <CheckCircle2 size={15} /> : <Ban size={15} />}
                        {status.charAt(0).toUpperCase() + status.slice(1)}
                      </button>
                    ))}
                  </div>

                  {approvalsError && (
                    <div className="info-card" role="alert">
                      <ShieldCheck size={19} />
                      <div><strong>Approval queue error</strong><p>{approvalsError}</p>
                        <button className="text-button" type="button" onClick={() => setApprovalsReload((current) => current + 1)}>Try again <ArrowRight size={14} /></button>
                      </div>
                    </div>
                  )}

                  {approvalsLoading && <div className="empty-state" role="status">Loading approval records…</div>}

                  {!approvalsLoading && !approvalsError && (
                    <>
                      <div className="section-heading documents-heading">
                        <div>
                          <h2>{approvals.length} {approvals.length === 1 ? 'record' : 'records'}</h2>
                          <p>Real approval records from the NexaGraph PostgreSQL database.</p>
                        </div>
                      </div>

                      {approvals.length === 0 ? (
                        <div className="info-card">
                          <FileCheck2 size={19} />
                          <div><strong>No {approvalStatus} approvals</strong><p>There are currently no records in this status.</p></div>
                        </div>
                      ) : (
                        <div className="review-list">
                          {approvals.map((approval) => (
                            <article className="review-row approval-card" key={approval.approval_id}>
                              <div className="review-icon blue"><FileCheck2 size={19} /></div>
                              <div className="review-content approval-content">
                                <div className="approval-heading">
                                  <strong>Approval #{approval.approval_id}</strong>
                                  <span className={`approval-status ${approval.approval_status}`}>
                                    {approval.approval_status}
                                  </span>
                                </div>

                                <div className="approval-meta">
                                  <span>Project: {approval.project_id}</span>
                                  <span>Request: {approval.request_id}</span>
                                  <span>Submitted: {formatDate(approval.created_at)}</span>
                                </div>

                                <div className="approval-question">
                                  <span className="approval-section-label">EMPLOYEE QUESTION</span>
                                  <p>{approval.employee_query}</p>
                                </div>

                                <div className="approval-answer">
                                  <span className="approval-section-label">AI-GENERATED ANSWER</span>
                                  <p>{approval.ai_answer}</p>
                                </div>

                                <div className="approval-meta">
                                  <span>Retrieval: {approval.retrieval_method || 'Not specified'}</span>
                                  {approval.reviewer_id && <span>Reviewer: {approval.reviewer_id}</span>}
                                  {approval.reviewed_at && <span>Reviewed: {formatDate(approval.reviewed_at)}</span>}
                                </div>

                                <div className="approval-citations">
                                  <span className="approval-section-label">CITATIONS ({approval.citations?.length ?? 0})</span>
                                  {approval.citations?.length ? (
                                    <div className="source-list">
                                      {approval.citations.map((citation, index) => (
                                        <div className="source-item" key={`${String(citation.citation_id ?? citation.document_id ?? index)}-${index}`}>
                                          <div className="source-number">{String(citation.citation_id ?? index + 1)}</div>
                                          <div className="source-body">
                                            <strong>{String(citation.source_file ?? citation.document_id ?? `Source ${index + 1}`)}</strong>
                                            <p>
                                              {[
                                                citation.page_number != null ? `Page ${String(citation.page_number)}` : '',
                                                citation.document_id != null ? `Document: ${String(citation.document_id)}` : '',
                                              ].filter(Boolean).join(' · ') || 'Citation metadata returned by the backend.'}
                                            </p>
                                          </div>
                                        </div>
                                      ))}
                                    </div>
                                  ) : (
                                    <p className="approval-muted">No citations were recorded for this answer.</p>
                                  )}
                                </div>

                                {approval.reviewer_comments && (
                                  <div className="approval-comments-display">
                                    <span className="approval-section-label">REVIEWER COMMENTS</span>
                                    <p>{approval.reviewer_comments}</p>
                                  </div>
                                )}

                                {approval.approval_status === 'pending' && (
                                  <div className="approval-actions">
                                    <label className="approval-comment-label" htmlFor={`comments-${approval.approval_id}`}>
                                      Reviewer comments (optional)
                                    </label>
                                    <textarea
                                      id={`comments-${approval.approval_id}`}
                                      value={approvalComments[approval.approval_id] ?? ''}
                                      onChange={(event) => setApprovalComments((current) => ({
                                        ...current,
                                        [approval.approval_id]: event.target.value,
                                      }))}
                                      placeholder="Add a reason or review note…"
                                      rows={2}
                                      maxLength={2000}
                                      disabled={decisionBusy === approval.approval_id} />
                                    <div className="approval-action-buttons">
                                      <button
                                        className="primary-button"
                                        type="button"
                                        disabled={decisionBusy !== null}
                                        onClick={() => void submitDecision(approval.approval_id, 'approved')}
                                      >
                                        <Check size={16} />
                                        {decisionBusy === approval.approval_id ? 'Saving…' : 'Approve'}
                                      </button>
                                      <button
                                        className="secondary-button"
                                        type="button"
                                        disabled={decisionBusy !== null}
                                        onClick={() => void submitDecision(approval.approval_id, 'rejected')}
                                      >
                                        <Ban size={16} /> Reject
                                      </button>
                                    </div>
                                  </div>
                                )}
                              </div>
                            </article>
                          ))}
                        </div>
                      )}
                    </>
                  )}
                </>
              )}
            </section>
          )}

          {activePage === 'Activity' && (
            <section className="page-section">
              <div className="page-title-row">
                <div>
                  <div className="eyebrow">WORKSPACE OVERVIEW</div>
                  <h1>Activity</h1>
                  <p>Recent actions recorded by the NexaGraph audit system.</p>
                </div>
                {isManager && (
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() => setActivityReload((current) => current + 1)}
                    disabled={activityLoading}
                  >
                    <RefreshCw size={15} /> {activityLoading ? 'Refreshing…' : 'Refresh'}
                  </button>
                )}
              </div>

              {!isManager ? (
                <div className="info-card">
                  <ShieldCheck size={19} />
                  <div>
                    <strong>Manager access required</strong>
                    <p>The activity feed is restricted to authenticated users with the manager role.</p>
                  </div>
                </div>
              ) : (
                <>
                  {activityError && (
                    <div className="info-card" role="alert">
                      <ShieldCheck size={19} />
                      <div>
                        <strong>Could not load activity</strong>
                        <p>{activityError}</p>
                        <button
                          className="text-button"
                          type="button"
                          onClick={() => setActivityReload((current) => current + 1)}
                        >
                          Try again <ArrowRight size={14} />
                        </button>
                      </div>
                    </div>
                  )}

                  {activityLoading && (
                    <div className="empty-state" role="status">Loading audit events…</div>
                  )}

                  {!activityLoading && !activityError && (
                    <>
                      <div className="activity-toolbar">
                        <label className="toolbar-search activity-search">
                          <Search size={16} />
                          <input
                            type="search"
                            value={activitySearch}
                            onChange={(event) => {
                              setActivitySearch(event.target.value)
                              setActivityPage(1)
                            }}
                            placeholder="Search action, user, project or ID…"
                            aria-label="Search activity"
                          />
                        </label>
                        <label className="activity-filter-label">
                          <span>Outcome</span>
                          <select
                            className="toolbar-filter activity-filter"
                            value={activityOutcome}
                            onChange={(event) => {
                              setActivityOutcome(event.target.value)
                              setActivityPage(1)
                            }}
                            aria-label="Filter activity by outcome"
                          >
                            <option value="all">All outcomes</option>
                            <option value="success">Success</option>
                            <option value="denied">Denied</option>
                            <option value="failure">Failure</option>
                          </select>
                        </label>
                      </div>

                      <div className="activity-table-heading">
                        <div>
                          <h2>{filteredActivityEvents.length} {filteredActivityEvents.length === 1 ? 'event' : 'events'}</h2>
                          <p>Audit records from PostgreSQL · Newest first</p>
                        </div>
                        <span className="activity-total">{activityEvents.length} loaded</span>
                      </div>

                      {filteredActivityEvents.length === 0 ? (
                        <div className="empty-state">
                          {activityEvents.length === 0
                            ? 'No activity has been recorded yet.'
                            : 'No activity matches your search or selected outcome.'}
                        </div>
                      ) : (
                        <>
                          <div className="activity-table-wrap">
                            <table className="activity-table">
                              <thead>
                                <tr>
                                  <th>Action</th>
                                  <th>User</th>
                                  <th>Project</th>
                                  <th>Date &amp; time</th>
                                  <th>Outcome</th>
                                  <th aria-label="Event details" />
                                </tr>
                              </thead>
                              <tbody>
                                {visibleActivityEvents.map((event) => (
                                  <Fragment key={event.event_id}>
                                    <tr className="activity-data-row">
                                      <td>
                                        <div className="activity-action-cell">
                                          <span className="activity-table-icon"><Activity size={15} /></span>
                                          <span>
                                            <strong>{event.event_type.replaceAll('_', ' ')}</strong>
                                            <small>Event #{event.event_id}</small>
                                          </span>
                                        </div>
                                      </td>
                                      <td>{event.user_id || 'System'}</td>
                                      <td>{event.project_id || '—'}</td>
                                      <td className="activity-date-cell">{formatDate(event.occurred_at)}</td>
                                      <td>
                                        <span className={`activity-outcome ${event.outcome.toLowerCase()}`}>
                                          <span />
                                          {event.outcome}
                                        </span>
                                      </td>
                                      <td>
                                        <button
                                          className="activity-expand-button"
                                          type="button"
                                          aria-expanded={expandedActivityId === event.event_id}
                                          onClick={() => setExpandedActivityId(
                                            expandedActivityId === event.event_id ? null : event.event_id,
                                          )}
                                        >
                                          {expandedActivityId === event.event_id ? 'Hide' : 'Details'}
                                          <ChevronDown size={14} />
                                        </button>
                                      </td>
                                    </tr>
                                    {expandedActivityId === event.event_id && (
                                      <tr key={`${event.event_id}-details`} className="activity-detail-row">
                                        <td colSpan={6}>
                                          <div className="activity-detail-grid">
                                            <div><span>Resource ID</span><strong>{event.resource_id || '—'}</strong></div>
                                            <div><span>Request ID</span><strong>{event.request_id || '—'}</strong></div>
                                            <div className="activity-json-detail">
                                              <span>Event details</span>
                                              <pre>{JSON.stringify(event.details || {}, null, 2)}</pre>
                                            </div>
                                          </div>
                                        </td>
                                      </tr>
                                    )}
                                  </Fragment>
                                ))}
                              </tbody>
                            </table>
                          </div>

                          <div className="activity-pagination">
                            <span>Showing {(activityPage - 1) * activityPageSize + 1}–{Math.min(activityPage * activityPageSize, filteredActivityEvents.length)} of {filteredActivityEvents.length}</span>
                            <div>
                              <button
                                type="button"
                                className="secondary-button"
                                onClick={() => setActivityPage((page) => Math.max(1, page - 1))}
                                disabled={activityPage <= 1}
                              >
                                Previous
                              </button>
                              <span>Page {activityPage} of {activityPageCount}</span>
                              <button
                                type="button"
                                className="secondary-button"
                                onClick={() => setActivityPage((page) => Math.min(activityPageCount, page + 1))}
                                disabled={activityPage >= activityPageCount}
                              >
                                Next
                              </button>
                            </div>
                          </div>
                        </>
                      )}
                    </>
                  )}
                </>
              )}
            </section>
          )}

          {activePage === 'Settings' && (
            <section className="page-section">
              <div className="page-title-row"><div><div className="eyebrow">ACCOUNT</div><h1>Settings</h1><p>Current authenticated account information.</p></div></div>
              <div className="settings-list">
                <div className="settings-row"><div className="settings-icon"><Users size={19} /></div><div><strong>Employee ID</strong><p>{user.user_id}</p></div><span className="settings-tag">Verified</span></div>
                <div className="settings-row"><div className="settings-icon"><LockKeyhole size={19} /></div><div><strong>Role</strong><p>{user.role}</p></div><span className="settings-tag">Backend</span></div>
                <div className="settings-row"><div className="settings-icon"><ShieldCheck size={19} /></div><div><strong>Authorized projects</strong><p>{projects.map((item) => item.name).join(', ')}</p></div><span className="settings-tag">Enforced</span></div>
              </div>
            </section>
          )}

          <footer className="main-footer"><div className="footer-brand"><Layers3 size={15} /> NexaGraph <span>·</span> Knowledge intelligence</div><div><span className="footer-status-dot" /> Authenticated environment <span className="footer-divider">|</span> API-connected</div></footer>
        </div>
      </main>
    </div>
  )
}

export default App
