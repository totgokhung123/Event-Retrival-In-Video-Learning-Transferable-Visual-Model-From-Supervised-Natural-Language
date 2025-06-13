import React from 'react';
import { Sidebar } from './components/Sidebar';
import { MainContent } from './components/MainContent';
import { VideoProvider } from './context/VideoContext';

function App() {
  return (
    <VideoProvider>
      <div className="flex h-screen bg-slate-900 text-slate-200 overflow-hidden">
        <Sidebar />
        <MainContent />
      </div>
    </VideoProvider>
  );
}

export default App;