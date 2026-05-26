import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'

// 同时支持 dev（vite dev server，挂 #card-forge-root）和 IIFE 内嵌主应用
// （主应用模板里准备一个 #card-forge-root 容器）。
function mount() {
  const container = document.getElementById('card-forge-root')
  if (!container) {
    console.error('[card-forge] mount target #card-forge-root not found')
    return
  }
  ReactDOM.createRoot(container).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  )
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', mount, { once: true })
} else {
  mount()
}
