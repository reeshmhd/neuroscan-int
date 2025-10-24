import React from 'react'
import { Routes,Route } from 'react-router-dom'
import Index from './pages/Index'
import Upload from './pages/Upload'
import Results from './pages/Results'
import NotFound from './pages/NotFound'

export default function App(){
  return (
    <Routes>
      <Route path="/" element={<Index/>}/>
      <Route path="/upload" element={<Upload/>}/>
      <Route path="/results" element={<Results/>}/>
      <Route path="*" element={<NotFound/>}/>
    </Routes>
  )
}
