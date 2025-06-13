import React from 'react';
import { VideoPlayer } from './VideoPlayer';
import { Timeline } from './Timeline';
import { AdvancedSearchPanel } from './AdvancedSearchPanel';
import { EventsSection } from './EventsSection';
import { useVideo } from '../context/VideoContext';

export const MainContent = () => {
  const { currentVideo } = useVideo();

  return (
    <div className="flex-1 h-screen flex flex-col overflow-hidden">
      <div className="p-4 border-b border-slate-700 flex justify-between items-center flex-shrink-0">
        <div>
          <h1 className="text-xl font-semibold">
            {currentVideo ? currentVideo.title : 'Video Event Query'}
          </h1>
          {currentVideo && (
            <p className="text-sm text-slate-400">
              {currentVideo.resolution} • {currentVideo.size}
            </p>
          )}
        </div>
      </div>

      <div className="flex-1 flex min-h-0">
        <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
          {/* Scrollable main content area */}
          <div className="flex-1 overflow-y-auto scrollbar-hide">
            <div className="min-h-full flex flex-col">
              {/* Video Player Section */}
              <div className="flex-shrink-0 p-6">
                <div className="h-[60vh] min-h-[400px]">
                  <VideoPlayer />
                </div>
              </div>
              
              {/* Timeline Section */}
              <div className="flex-shrink-0 px-6 pb-4">
                <div className="h-32">
                  <Timeline />
                </div>
              </div>
              
              {/* Events Section */}
              <div className="flex-1 min-h-[400px]">
                <EventsSection />
              </div>
            </div>
          </div>
        </div>
        
        {/* Right Panel */}
        <div className="w-80 border-l border-slate-700 overflow-y-auto scrollbar-hide">
          <AdvancedSearchPanel />
        </div>
      </div>
    </div>
  );
};