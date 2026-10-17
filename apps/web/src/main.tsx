import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import CommunityWorkspace from './CommunityWorkspace.tsx'
import { desktopStudioApplication } from './desktop.ts'
import ReviewWorkspace from './ReviewWorkspace.tsx'

const desktopApplication = desktopStudioApplication()
const rootView =
  window.location.pathname === '/review' ? (
    <ReviewWorkspace />
  ) : window.location.pathname === '/community' ? (
    <CommunityWorkspace />
  ) : (
    <App application={desktopApplication ?? undefined} />
  )

createRoot(document.getElementById('root')!).render(
  <StrictMode>{rootView}</StrictMode>,
)
