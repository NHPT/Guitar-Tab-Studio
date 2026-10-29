import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import { desktopStudioApplication } from './desktop.ts'
import './index.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App application={desktopStudioApplication() ?? undefined} />
  </StrictMode>,
)
