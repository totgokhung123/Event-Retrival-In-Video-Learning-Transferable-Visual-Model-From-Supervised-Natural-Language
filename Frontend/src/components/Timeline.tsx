import React, { useState, useRef } from 'react';
import { useVideo } from '../context/VideoContext';
import { formatDuration } from '../utils/formatters';

export const Timeline = () => {
  const { currentVideo, events, currentTime, setCurrentTime } = useVideo();
  const [isDragging, setIsDragging] = useState(false);
  const timelineRef = useRef<HTMLDivElement>(null);

  if (!currentVideo) {
    return (
      <div className="w-full h-full bg-slate-800 rounded-lg flex items-center justify-center">
        <p className="text-slate-500 text-sm">No video selected</p>
      </div>
    );
  }

  const handleTimelineClick = (e: React.MouseEvent) => {
    if (!timelineRef.current) return;
    
    const rect = timelineRef.current.getBoundingClientRect();
    const percentage = (e.clientX - rect.left) / rect.width;
    const newTime = percentage * currentVideo.duration;
    setCurrentTime(Math.max(0, Math.min(newTime, currentVideo.duration)));
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    handleTimelineClick(e);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      handleTimelineClick(e);
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Filter events for current video
  const videoEvents = events.filter((event) => event.videoId === currentVideo.id);

  // Calculate current progress percentage
  const progressPercentage = (currentTime / currentVideo.duration) * 100;

  return (
    <div className="w-full h-full flex flex-col bg-slate-800 rounded-lg p-3">
      <div className="flex justify-between items-center mb-2">
        <h3 className="text-sm font-medium">Timeline</h3>
        <span className="text-xs text-slate-400">
          {formatDuration(currentTime)} / {formatDuration(currentVideo.duration)}
        </span>
      </div>

      <div
        ref={timelineRef}
        className="relative h-12 bg-slate-700 rounded-md cursor-pointer"
        onClick={handleTimelineClick}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        {/* Progress bar */}
        <div
          className="absolute top-0 left-0 h-full bg-slate-600 rounded-l-md"
          style={{ width: `${progressPercentage}%` }}
        ></div>

        {/* Current position indicator */}
        <div
          className="absolute top-0 w-0.5 h-full bg-teal-500 z-10"
          style={{ left: `${progressPercentage}%` }}
        >
          <div className="absolute -left-1 top-0 w-2 h-2 rounded-full bg-teal-500"></div>
          <div className="absolute -left-1 bottom-0 w-2 h-2 rounded-full bg-teal-500"></div>
        </div>

        {/* Event markers */}
        {videoEvents.map((event) => {
          const startPercent = (event.timestamp / currentVideo.duration) * 100;
          const widthPercent = (event.duration / currentVideo.duration) * 100;
          
          return (
            <div
              key={event.id}
              className="absolute top-0 h-full bg-amber-500/40 hover:bg-amber-500/60 transition-colors duration-200"
              style={{
                left: `${startPercent}%`,
                width: `${widthPercent}%`,
              }}
              title={event.title}
              onClick={(e) => {
                e.stopPropagation();
                setCurrentTime(event.timestamp);
              }}
            ></div>
          );
        })}

        {/* Time intervals */}
        <div className="absolute bottom-0 left-0 right-0 flex justify-between px-2 text-xs text-slate-400">
          <span>0:00</span>
          <span>{formatDuration(currentVideo.duration / 4)}</span>
          <span>{formatDuration(currentVideo.duration / 2)}</span>
          <span>{formatDuration((currentVideo.duration / 4) * 3)}</span>
          <span>{formatDuration(currentVideo.duration)}</span>
        </div>
      </div>

      <div className="mt-1 text-xs text-slate-400">
        <span>{videoEvents.length} events detected</span>
      </div>
    </div>
  );
};