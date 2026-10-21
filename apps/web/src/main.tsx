import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import AdminWorkspace from './AdminWorkspace.tsx'
import App from './App.tsx'
import CommunityWorkspace from './CommunityWorkspace.tsx'
import { desktopStudioApplication } from './desktop.ts'
import ReviewWorkspace from './ReviewWorkspace.tsx'

const desktopApplication = desktopStudioApplication()
if (window.location.pathname === '/admin/beat-review') {
  window.history.replaceState(
    null,
    '',
    `/admin/reviews/beat${window.location.search}`,
  )
}
const rootView =
  window.location.pathname.startsWith('/admin') ? (
    <AdminWorkspace />
  ) : window.location.pathname === '/review' ? (
    <ReviewWorkspace />
  ) : window.location.pathname === '/community' ? (
    <CommunityWorkspace />
  ) : (
    <App application={desktopApplication ?? undefined} />
  )

createRoot(document.getElementById('root')!).render(
  <StrictMode>{rootView}</StrictMode>,
)
