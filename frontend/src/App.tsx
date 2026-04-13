import { useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  Bot,
  FileText,
  Menu,
  PanelRight,
  Plus,
  Search,
  SendHorizonal,
  Sparkles,
  Trash2,
  User,
  X,
} from 'lucide-react'

type SourceItem = {
  source: string
  section: string
  score: number
  content?: string
}

type ApiResponse = {
  answer: string
  sources: SourceItem[]
}

type Message = {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: SourceItem[]
  createdAt: number
}

type ChatSession = {
  id: string
  title: string
  preview: string
  messages: Message[]
  updatedAt: number
}

const exampleQuestions = [
  '大创项目申报需要满足哪些基本条件？',
  '国家级、省级和校级大创项目有什么区别？',
  '大创项目团队成员人数和指导老师有哪些要求？',
  '大创项目中期检查和结题验收通常需要提交哪些材料？',
]

const initialGreeting = `你好，我是你的规章制度智能助手。\n\n你可以直接问我制度要求、流程步骤、适用条件、时间节点或责任分工。\n\n我会结合检索到的文档来源，为你提供带引用的回答。`
const SESSION_STORAGE_KEY = 'rag_chat_sessions_v1'

const createInitialSession = (): ChatSession => ({
  id: crypto.randomUUID(),
  title: '新对话',
  preview: '开始提问，获取制度解答',
  updatedAt: Date.now(),
  messages: [
    {
      id: crypto.randomUUID(),
      role: 'assistant',
      content: initialGreeting,
      createdAt: Date.now(),
      sources: [],
    },
  ],
})

function loadSessionsFromStorage(): ChatSession[] {
  if (typeof window === 'undefined') {
    return [createInitialSession()]
  }

  try {
    const raw = localStorage.getItem(SESSION_STORAGE_KEY)
    if (!raw) {
      return [createInitialSession()]
    }

    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed) || !parsed.length) {
      return [createInitialSession()]
    }

    const normalized = parsed.filter(
      (item) => item?.id && item?.title && Array.isArray(item?.messages),
    ) as ChatSession[]

    return normalized.length ? normalized : [createInitialSession()]
  } catch {
    return [createInitialSession()]
  }
}

function formatTime(timestamp: number) {
  return new Intl.DateTimeFormat('zh-CN', {
    hour: '2-digit',
    minute: '2-digit',
  }).format(timestamp)
}

function clamp(value: number, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value))
}

function getSessionTitle(question: string) {
  const normalized = question.trim().replace(/\s+/g, ' ')
  return normalized.length > 18 ? `${normalized.slice(0, 18)}...` : normalized
}

function getScoreWidth(score: number) {
  return `${Math.round(clamp(score) * 100)}%`
}

function TypewriterMarkdown({
  content,
  active,
}: {
  content: string
  active: boolean
}) {
  const [displayed, setDisplayed] = useState(active ? '' : content)

  useEffect(() => {
    if (!active) {
      setDisplayed(content)
      return
    }

    setDisplayed('')
    let index = 0
    const timer = window.setInterval(() => {
      index += 2
      setDisplayed(content.slice(0, index))
      if (index >= content.length) {
        window.clearInterval(timer)
      }
    }, 12)

    return () => window.clearInterval(timer)
  }, [content, active])

  return (
    <div className="markdown-body">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{displayed}</ReactMarkdown>
      {active && (
        <motion.span
          className="ml-1 inline-block h-5 w-[2px] rounded-full bg-cyan-300 align-middle"
          animate={{ opacity: [0.25, 1, 0.25] }}
          transition={{ duration: 1, repeat: Infinity }}
        />
      )}
    </div>
  )
}

function MessageBubble({
  message,
  isStreaming,
}: {
  message: Message
  isStreaming: boolean
}) {
  const isUser = message.role === 'user'

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex gap-4 ${isUser ? 'justify-end' : 'justify-start'}`}
    >
      {!isUser && (
        <div className="mt-1 flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl border border-white/10 bg-gradient-to-br from-cyan-400/30 to-violet-500/30 backdrop-blur-xl">
          <Bot className="h-5 w-5 text-cyan-100" />
        </div>
      )}

      <div className={`flex max-w-[88%] flex-col ${isUser ? 'items-end' : 'items-start'}`}>
        <div className="mb-2 flex items-center gap-2 px-1 text-xs text-slate-400">
          <span className="font-medium text-slate-300">{isUser ? '你' : 'AI 助手'}</span>
          <span>{formatTime(message.createdAt)}</span>
        </div>

        <div
          className={[
            'rounded-[28px] border px-5 py-4 shadow-2xl backdrop-blur-xl',
            isUser
              ? 'border-cyan-400/20 bg-gradient-to-br from-cyan-500/20 to-blue-500/10 text-slate-100'
              : 'border-white/10 bg-white/8 text-slate-100',
          ].join(' ')}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap leading-7 text-slate-100">{message.content}</p>
          ) : (
            <TypewriterMarkdown content={message.content} active={isStreaming} />
          )}
        </div>
      </div>

      {isUser && (
        <div className="mt-1 flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl border border-white/10 bg-white/8 backdrop-blur-xl">
          <User className="h-5 w-5 text-slate-200" />
        </div>
      )}
    </motion.div>
  )
}

function SidebarContent({
  sessions,
  activeSessionId,
  setActiveSessionId,
  createNewChat,
  clearAllHistory,
  mobile = false,
  closeMobile,
}: {
  sessions: ChatSession[]
  activeSessionId: string
  setActiveSessionId: (id: string) => void
  createNewChat: () => void
  clearAllHistory: () => void
  mobile?: boolean
  closeMobile?: () => void
}) {
  return (
    <>
      <div className="mb-4 flex items-center justify-between">
        <div className="flex gap-2">
          <button
            onClick={createNewChat}
            className="inline-flex items-center gap-2 rounded-2xl border border-cyan-400/20 bg-gradient-to-r from-cyan-500/25 to-violet-500/25 px-4 py-3 text-sm font-medium text-cyan-50"
          >
            <Plus className="h-4 w-4" />
            新建对话
          </button>
          <button
            onClick={clearAllHistory}
            className="inline-flex items-center gap-2 rounded-2xl border border-white/15 bg-white/10 px-3 py-3 text-sm text-slate-200 transition hover:bg-white/15"
            title="清空历史"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
        {mobile && closeMobile && (
          <button
            onClick={closeMobile}
            className="rounded-xl border border-white/10 p-2 text-slate-300"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      <div className="mb-4 flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-3 py-2 text-slate-400">
        <Search className="h-4 w-4" />
        <span className="text-sm">历史会话</span>
      </div>

      <div className="space-y-3">
        {sessions
          .slice()
          .sort((a, b) => b.updatedAt - a.updatedAt)
          .map((session) => {
            const isActive = session.id === activeSessionId
            return (
              <button
                key={session.id}
                onClick={() => setActiveSessionId(session.id)}
                className={`w-full rounded-2xl border p-4 text-left transition ${
                  isActive
                    ? 'border-cyan-400/30 bg-gradient-to-br from-cyan-500/15 to-violet-500/15'
                    : 'border-white/8 bg-white/5 hover:bg-white/8'
                }`}
              >
                <div className="mb-1 line-clamp-1 text-sm font-medium text-slate-100">{session.title}</div>
                <div className="line-clamp-2 text-xs leading-5 text-slate-400">{session.preview}</div>
                <div className="mt-3 text-[11px] text-slate-500">{formatTime(session.updatedAt)}</div>
              </button>
            )
          })}
      </div>
    </>
  )
}

function SourcesPanel({ sources }: { sources: SourceItem[] }) {
  return (
    <div>
      <div className="mb-4">
        <div className="text-xs uppercase tracking-[0.26em] text-slate-500">Retrieval Sources</div>
        <h2 className="mt-1 text-lg font-semibold text-slate-100">引用来源</h2>
      </div>

      <div className="space-y-3">
        {sources.length ? (
          sources.map((item, index) => (
            <motion.div
              key={`${item.source}-${item.section}-${index}`}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              className="rounded-3xl border border-white/10 bg-white/5 p-4"
            >
              <div className="mb-3 flex items-start gap-3">
                <div className="mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-cyan-400/20 to-violet-500/20 text-cyan-100">
                  <FileText className="h-4 w-4" />
                </div>
                <div className="min-w-0">
                  <div className="truncate text-sm font-medium text-slate-100">{item.source}</div>
                  <div className="mt-1 text-xs leading-5 text-slate-400">{item.section || '未标注章节'}</div>
                </div>
              </div>

              <div className="mb-2 flex items-center justify-between text-xs text-slate-400">
                <span>相关度</span>
                <span className="font-medium text-cyan-200">{(clamp(item.score) * 100).toFixed(0)}%</span>
              </div>

              <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: getScoreWidth(item.score) }}
                  className="h-full rounded-full bg-gradient-to-r from-cyan-400 via-blue-500 to-violet-500"
                />
              </div>

              {item.content ? (
                <p className="mt-3 line-clamp-3 text-xs leading-6 text-slate-400">{item.content}</p>
              ) : null}
            </motion.div>
          ))
        ) : (
          <div className="rounded-3xl border border-dashed border-white/10 bg-white/[0.03] p-6 text-sm leading-7 text-slate-400">
            当前还没有可展示的引用来源。<br />发送问题后，这里会展示检索到的文档、章节和相关度分数。
          </div>
        )}
      </div>
    </div>
  )
}

function App() {
  const initialSessions = useMemo(() => loadSessionsFromStorage(), [])
  const [sessions, setSessions] = useState<ChatSession[]>(initialSessions)
  const [activeSessionId, setActiveSessionId] = useState<string>(initialSessions[0].id)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null)
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false)
  const [mobileSourcesOpen, setMobileSourcesOpen] = useState(false)
  const chatEndRef = useRef<HTMLDivElement | null>(null)

  const activeSession = useMemo(
    () => sessions.find((session) => session.id === activeSessionId) ?? sessions[0],
    [sessions, activeSessionId],
  )

  const latestSources = useMemo(() => {
    const messages = activeSession?.messages ?? []
    return (
      [...messages]
        .reverse()
        .find((message) => message.role === 'assistant' && (message.sources?.length ?? 0) > 0)
        ?.sources ?? []
    )
  }, [activeSession])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [activeSession?.messages, loading])

  useEffect(() => {
    localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(sessions))
  }, [sessions])

  useEffect(() => {
    const exists = sessions.some((session) => session.id === activeSessionId)
    if (!exists) {
      setActiveSessionId(sessions[0].id)
    }
  }, [sessions, activeSessionId])

  const updateSession = (sessionId: string, updater: (session: ChatSession) => ChatSession) => {
    setSessions((prev) =>
      prev.map((session) => (session.id === sessionId ? updater(session) : session)),
    )
  }

  const createNewChat = () => {
    const newSession = createInitialSession()
    setSessions((prev) => [newSession, ...prev])
    setActiveSessionId(newSession.id)
    setMobileSidebarOpen(false)
    setMobileSourcesOpen(false)
  }

  const clearAllHistory = () => {
    const fresh = createInitialSession()
    setSessions([fresh])
    setActiveSessionId(fresh.id)
    setMobileSidebarOpen(false)
    setMobileSourcesOpen(false)
  }

  const sendQuestion = async (question: string) => {
    const trimmed = question.trim()
    if (!trimmed || loading || !activeSession) return

    const sessionId = activeSession.id
    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: trimmed,
      createdAt: Date.now(),
    }

    const assistantMessageId = crypto.randomUUID()
    const assistantPlaceholder: Message = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      createdAt: Date.now(),
      sources: [],
    }

    const nextTitle =
      activeSession.messages.length <= 1 ? getSessionTitle(trimmed) : activeSession.title

    updateSession(sessionId, (session) => ({
      ...session,
      title: nextTitle,
      preview: trimmed,
      updatedAt: Date.now(),
      messages: [...session.messages, userMessage, assistantPlaceholder],
    }))

    setInput('')
    setLoading(true)
    setStreamingMessageId(assistantMessageId)

    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmed }),
      })

      if (!response.ok) {
        throw new Error(`请求失败: ${response.status}`)
      }

      const data: ApiResponse = await response.json()

      updateSession(sessionId, (session) => ({
        ...session,
        updatedAt: Date.now(),
        messages: session.messages.map((message) =>
          message.id === assistantMessageId
            ? {
                ...message,
                content: data.answer || '未获取到回答。',
                sources: data.sources || [],
              }
            : message,
        ),
      }))
    } catch (error) {
      const errorText =
        error instanceof Error
          ? `抱歉，当前请求失败。\n\n错误信息：${error.message}`
          : '抱歉，当前请求失败，请稍后重试。'

      updateSession(sessionId, (session) => ({
        ...session,
        updatedAt: Date.now(),
        messages: session.messages.map((message) =>
          message.id === assistantMessageId
            ? { ...message, content: errorText, sources: [] }
            : message,
        ),
      }))
    } finally {
      setLoading(false)
      window.setTimeout(() => setStreamingMessageId(null), 600)
    }
  }

  const handleSubmit = async () => {
    await sendQuestion(input)
  }

  const handleKeyDown = async (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      await handleSubmit()
    }
  }

  return (
    <div className="relative min-h-screen overflow-hidden bg-[#0b1a2d] text-slate-100">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.22),transparent_30%),radial-gradient(circle_at_top_right,rgba(99,102,241,0.2),transparent_30%),radial-gradient(circle_at_bottom,rgba(45,212,191,0.12),transparent_35%)]" />
      <div className="absolute inset-0 opacity-45 [background-image:linear-gradient(rgba(148,163,184,0.1)_1px,transparent_1px),linear-gradient(90deg,rgba(148,163,184,0.1)_1px,transparent_1px)] [background-size:48px_48px]" />

      <div className="relative flex min-h-screen">
        <AnimatePresence>
          {mobileSidebarOpen ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-30 bg-slate-950/55 lg:hidden"
              onClick={() => setMobileSidebarOpen(false)}
            />
          ) : null}
        </AnimatePresence>

        <aside className="hidden w-[290px] border-r border-white/10 bg-slate-950/55 p-4 backdrop-blur-2xl lg:block">
          <SidebarContent
            sessions={sessions}
            activeSessionId={activeSessionId}
            setActiveSessionId={setActiveSessionId}
            createNewChat={createNewChat}
            clearAllHistory={clearAllHistory}
          />
        </aside>

        <AnimatePresence>
          {mobileSidebarOpen ? (
            <motion.aside
              initial={{ x: -24, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: -24, opacity: 0 }}
              className="fixed inset-y-0 left-0 z-40 w-[290px] border-r border-white/10 bg-slate-950/95 p-4 backdrop-blur-2xl lg:hidden"
            >
              <SidebarContent
                sessions={sessions}
                activeSessionId={activeSessionId}
                setActiveSessionId={(id) => {
                  setActiveSessionId(id)
                  setMobileSidebarOpen(false)
                }}
                createNewChat={createNewChat}
                clearAllHistory={clearAllHistory}
                mobile
                closeMobile={() => setMobileSidebarOpen(false)}
              />
            </motion.aside>
          ) : null}
        </AnimatePresence>

        <main className="flex min-w-0 flex-1 flex-col">
          <header className="sticky top-0 z-20 border-b border-white/10 bg-slate-950/35 px-4 py-4 backdrop-blur-2xl lg:px-8">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setMobileSidebarOpen(true)}
                  className="rounded-xl border border-white/10 bg-white/5 p-2 text-slate-300 lg:hidden"
                >
                  <Menu className="h-5 w-5" />
                </button>

                <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-cyan-400/20 bg-gradient-to-br from-cyan-400/20 via-blue-500/15 to-violet-500/25 shadow-[0_0_24px_rgba(59,130,246,0.22)]">
                  <Sparkles className="h-5 w-5 text-cyan-100" />
                </div>

                <div>
                  <div className="text-[11px] uppercase tracking-[0.32em] text-cyan-300/80">
                    AI Knowledge Assistant
                  </div>
                  <h1 className="text-lg font-semibold text-slate-50">规章制度智能问答</h1>
                </div>
              </div>

              <button
                onClick={() => setMobileSourcesOpen(true)}
                className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-200 lg:hidden"
              >
                <PanelRight className="h-4 w-4" />
                来源
              </button>
            </div>
          </header>

          <div className="flex min-h-0 flex-1">
            <section className="flex min-w-0 flex-1 flex-col px-4 pb-4 pt-6 lg:px-8 lg:pb-6">
              <div className="mx-auto flex w-full max-w-5xl flex-1 flex-col">
                <div className="mb-4 flex flex-wrap gap-2">
                  {exampleQuestions.map((question) => (
                    <motion.button
                      key={question}
                      whileHover={{ y: -2 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => void sendQuestion(question)}
                      className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-300 transition hover:border-cyan-400/20 hover:bg-cyan-400/10 hover:text-cyan-100"
                    >
                      {question}
                    </motion.button>
                  ))}
                </div>

                <div className="flex-1 space-y-6 overflow-y-auto rounded-[32px] border border-cyan-200/15 bg-white/[0.09] p-4 shadow-[0_20px_80px_rgba(2,6,23,0.45)] backdrop-blur-xl md:p-6">
                  {activeSession?.messages.map((message) => (
                    <MessageBubble
                      key={message.id}
                      message={message}
                      isStreaming={streamingMessageId === message.id}
                    />
                  ))}
                  <div ref={chatEndRef} />
                </div>

                <div className="mt-4 rounded-[28px] border border-cyan-200/20 bg-slate-900/55 p-3 shadow-[0_20px_80px_rgba(15,23,42,0.35)] backdrop-blur-2xl">
                  <div className="rounded-[24px] border border-white/20 bg-white/12 px-4 py-3 focus-within:border-cyan-300/50 focus-within:bg-white/15">
                    <textarea
                      value={input}
                      onChange={(event) => setInput(event.target.value)}
                      onKeyDown={(event) => void handleKeyDown(event)}
                      rows={1}
                      placeholder="输入你的问题，按 Enter 发送，Shift + Enter 换行"
                      className="min-h-[56px] w-full resize-none bg-transparent text-sm leading-7 text-slate-50 outline-none placeholder:text-slate-300/70"
                    />
                  </div>

                  <div className="mt-3 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                    <div className="text-xs text-slate-400">支持 Markdown、引用来源和快捷提问</div>

                    <motion.button
                      whileTap={{ scale: 0.98 }}
                      whileHover={{ y: -1 }}
                      onClick={() => void handleSubmit()}
                      disabled={loading || !input.trim()}
                      className="inline-flex items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-cyan-500 to-violet-500 px-5 py-3 text-sm font-medium text-white shadow-[0_12px_30px_rgba(59,130,246,0.28)] transition disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {loading ? (
                        <motion.span
                          className="h-4 w-4 rounded-full border-2 border-white/30 border-t-white"
                          animate={{ rotate: 360 }}
                          transition={{ duration: 0.8, repeat: Infinity, ease: 'linear' }}
                        />
                      ) : (
                        <SendHorizonal className="h-4 w-4" />
                      )}
                      {loading ? '发送中' : '发送问题'}
                    </motion.button>
                  </div>
                </div>
              </div>
            </section>

            <aside className="hidden w-[360px] border-l border-white/10 bg-slate-950/55 p-4 backdrop-blur-2xl lg:block">
              <SourcesPanel sources={latestSources} />
            </aside>

            <AnimatePresence>
              {mobileSourcesOpen ? (
                <>
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="fixed inset-0 z-30 bg-slate-950/55 lg:hidden"
                    onClick={() => setMobileSourcesOpen(false)}
                  />
                  <motion.aside
                    initial={{ x: 24, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    exit={{ x: 24, opacity: 0 }}
                    className="fixed inset-y-0 right-0 z-40 w-full max-w-[360px] border-l border-white/10 bg-slate-950/95 p-4 backdrop-blur-2xl lg:hidden"
                  >
                    <div className="mb-4 flex items-center justify-between">
                      <h2 className="text-lg font-semibold text-slate-100">引用来源</h2>
                      <button
                        onClick={() => setMobileSourcesOpen(false)}
                        className="rounded-xl border border-white/10 p-2 text-slate-300"
                      >
                        <X className="h-4 w-4" />
                      </button>
                    </div>
                    <SourcesPanel sources={latestSources} />
                  </motion.aside>
                </>
              ) : null}
            </AnimatePresence>
          </div>
        </main>
      </div>
    </div>
  )
}

export default App
