import React, { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { MainContent } from './components/MainContent';
import { VideoProvider } from './context/VideoContext';

function App() {
  const [activeTab, setActiveTab] = useState('library');
  
  return (
    <VideoProvider>
      <div className="flex h-screen bg-slate-900 text-slate-200 overflow-hidden">
        <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />
        <MainContent activeTab={activeTab} />
      </div>
    </VideoProvider>
  );
}

export default App;