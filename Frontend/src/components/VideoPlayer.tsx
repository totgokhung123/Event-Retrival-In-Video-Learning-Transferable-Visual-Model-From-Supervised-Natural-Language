import React, { useRef, useEffect } from 'react';
import { Play, Pause, SkipBack, SkipForward, Maximize } from 'lucide-react';
import { useVideo } from '../context/VideoContext';
import { formatDuration } from '../utils/formatters';

const API_BASE_URL = 'http://localhost:5000/api';

export const VideoPlayer = () => {
  const {
    currentVideo,
    currentTime,
    isPlaying,
    setCurrentTime,
    setIsPlaying,
  } = useVideo();
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.play().catch((e) => {
          console.error('Error playing video:', e);
          setIsPlaying(false);
        });
      } else {
        videoRef.current.pause();
      }
    }
  }, [isPlaying, setIsPlaying]);

  useEffect(() => {
    if (videoRef.current && Math.abs(videoRef.current.currentTime - currentTime) > 0.5) {
      videoRef.current.currentTime = currentTime;
    }
  }, [currentTime]);

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleSkipForward = () => {
    if (videoRef.current) {
      videoRef.current.currentTime += 10;
    }
  };

  const handleSkipBackward = () => {
    if (videoRef.current) {
      videoRef.current.currentTime -= 10;
    }
  };

  const handleFullscreen = () => {
    if (videoRef.current) {
      if (document.fullscreenElement) {
        document.exitFullscreen();
      } else {
        videoRef.current.requestFullscreen();
      }
    }
  };

  // Xây dựng URL video từ backend
  const getVideoUrl = () => {
    if (!currentVideo) return '';
    if (currentVideo.path) {
      // Phân tích path để lấy tên file
      const parts = currentVideo.path.split(/[/\\]/);
      const filename = parts[parts.length - 1];
      return `${API_BASE_URL}/video/${filename}`;
    }
    return '';
  };

  if (!currentVideo) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-slate-800 rounded-lg">
        <div className="text-center p-8">
          <div className="mb-4 mx-auto w-16 h-16 rounded-full bg-slate-700 flex items-center justify-center">
            <Play size={24} className="text-slate-500 ml-1" />
          </div>
          <h3 className="text-xl font-medium mb-2">No Video Selected</h3>
          <p className="text-slate-400">
            Select a video from your library to begin analysis
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full flex items-center justify-center bg-black rounded-lg overflow-hidden group">
      <div className="w-full h-full relative">
        <video
          ref={videoRef}
          src={getVideoUrl()}
          poster={currentVideo.thumbnail ? `${API_BASE_URL}/frame/${currentVideo.thumbnail}` : undefined}
          className="w-full h-full object-contain"
          onTimeUpdate={handleTimeUpdate}
          onEnded={() => setIsPlaying(false)}
          crossOrigin="anonymous"
        />

        {/* Video Controls Overlay */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300">
          <div className="absolute bottom-0 left-0 right-0 p-4">
            <div className="flex items-center gap-3">
              <button
                onClick={handleSkipBackward}
                className="text-white hover:text-teal-400 transition-colors duration-200"
              >
                <SkipBack size={20} />
              </button>
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="w-10 h-10 bg-teal-500 hover:bg-teal-400 rounded-full flex items-center justify-center text-white transition-colors duration-200"
              >
                {isPlaying ? <Pause size={20} /> : <Play size={20} className="ml-1" />}
              </button>
              <button
                onClick={handleSkipForward}
                className="text-white hover:text-teal-400 transition-colors duration-200"
              >
                <SkipForward size={20} />
              </button>
              <div className="text-sm text-white ml-2">
                {formatDuration(currentTime)} / {formatDuration(currentVideo.duration)}
              </div>
              <div className="flex-1"></div>
              <button
                onClick={handleFullscreen}
                className="text-white hover:text-teal-400 transition-colors duration-200"
              >
                <Maximize size={20} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};