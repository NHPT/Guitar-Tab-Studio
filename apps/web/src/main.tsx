import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import CommunityWorkspace from './CommunityWorkspace.tsx'
import ReviewWorkspace from './ReviewWorkspace.tsx'

const RootView =
  window.location.pathname === '/review'
    ? ReviewWorkspace
    : window.location.pathname === '/community'
      ? CommunityWorkspace
      : App

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <RootView />
  </StrictMode>,
)
