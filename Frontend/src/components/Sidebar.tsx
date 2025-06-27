import React from 'react';
import { Menu, Video, Upload, History, Settings, Search, Zap } from 'lucide-react';
import { useVideo } from '../context/VideoContext';
import { VideoLibrary } from './VideoLibrary';

interface SidebarProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
}

export const Sidebar = ({ activeTab, onTabChange }: SidebarProps) => {
  const [expanded, setExpanded] = React.useState(true);
  const { videos } = useVideo();

  return (
    <div
      className={`flex flex-col bg-slate-800 h-full transition-all duration-300 ${
        expanded ? 'w-72' : 'w-16'
      }`}
    >
      <div className="flex items-center p-4 border-b border-slate-700">
        <button
          onClick={() => setExpanded(!expanded)}
          className="p-1 rounded-md hover:bg-slate-700 transition-colors duration-200"
        >
          <Menu size={20} />
        </button>
        {expanded && (
          <h1 className="ml-3 font-semibold text-lg">Video Event Query</h1>
        )}
      </div>

      <div className="flex flex-col flex-1 overflow-hidden">
        <nav className="flex flex-col gap-1 p-2">
          <SidebarButton
            icon={<Video size={20} />}
            label="Library"
            active={activeTab === 'library'}
            expanded={expanded}
            onClick={() => onTabChange('library')}
          />
          <SidebarButton
            icon={<Search size={20} />}
            label="Search"
            active={activeTab === 'search'}
            expanded={expanded}
            onClick={() => onTabChange('search')}
          />
          <SidebarButton
            icon={<Zap size={20} />}
            label="Visualization"
            active={activeTab === 'visualization'}
            expanded={expanded}
            onClick={() => onTabChange('visualization')}
          />
          <SidebarButton
            icon={<Upload size={20} />}
            label="Upload"
            active={activeTab === 'upload'}
            expanded={expanded}
            onClick={() => onTabChange('upload')}
          />
          <SidebarButton
            icon={<History size={20} />}
            label="History"
            active={activeTab === 'history'}
            expanded={expanded}
            onClick={() => onTabChange('history')}
          />
          <SidebarButton
            icon={<Settings size={20} />}
            label="Settings"
            active={activeTab === 'settings'}
            expanded={expanded}
            onClick={() => onTabChange('settings')}
          />
        </nav>

        {expanded && activeTab === 'library' && (
          <div className="flex-1 overflow-y-auto p-2">
            <VideoLibrary videos={videos} />
          </div>
        )}
      </div>

      {expanded && (
        <div className="p-4 border-t border-slate-700">
          <div className="flex items-center">
            <div className="w-8 h-8 rounded-full bg-teal-500 flex items-center justify-center">
              <span className="font-medium text-slate-900">AV</span>
            </div>
            <div className="ml-3">
              <p className="text-sm font-medium">Analyst View</p>
              <p className="text-xs text-slate-400">Advanced</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

interface SidebarButtonProps {
  icon: React.ReactNode;
  label: string;
  active: boolean;
  expanded: boolean;
  onClick: () => void;
}

const SidebarButton = ({
  icon,
  label,
  active,
  expanded,
  onClick,
}: SidebarButtonProps) => {
  return (
    <button
      onClick={onClick}
      className={`flex items-center p-2 rounded-md transition-colors duration-200 ${
        active
          ? 'bg-teal-600 text-white'
          : 'text-slate-400 hover:bg-slate-700 hover:text-slate-200'
      }`}
    >
      <span className="flex-shrink-0">{icon}</span>
      {expanded && <span className="ml-3">{label}</span>}
    </button>
  );
};