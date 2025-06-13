import React, { useState, useRef } from 'react';
import { File, Upload, Calendar, Film, X, Loader } from 'lucide-react';
import { useVideo } from '../context/VideoContext';
import { formatDuration } from '../utils/formatters';
import { VideoData } from '../types';

const API_BASE_URL = 'http://localhost:5000/api';

export const VideoLibrary = () => {
  const { videos, setCurrentVideo, currentVideo, isLoading, loadVideoEvents, uploadVideo } = useVideo();
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [showUpload, setShowUpload] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Xử lý chọn video
  const handleSelectVideo = (video: VideoData) => {
    setCurrentVideo(video);
    if (video && video.id) {
      loadVideoEvents(video.id);
    }
  };

  // Xử lý click nút upload
  const handleUploadClick = () => {
    setShowUpload(true);
  };

  // Xử lý đóng modal upload
  const handleCloseUpload = () => {
    setShowUpload(false);
  };

  // Xử lý khi file được chọn
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      setIsUploading(true);
      setUploadProgress(10); // Bắt đầu với 10%
      
      // Giả lập tiến trình upload
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 1000);
      
      // Upload file
      await uploadVideo(file);
      
      // Hoàn thành
      clearInterval(progressInterval);
      setUploadProgress(100);
      setTimeout(() => {
        setIsUploading(false);
        setUploadProgress(0);
        setShowUpload(false);
      }, 1000);
      
    } catch (error) {
      console.error('Upload failed:', error);
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  // Render thumbnail với URL từ backend
  const renderThumbnail = (video: VideoData) => {
    if (!video.thumbnail) {
      return (
        <div className="w-full h-full bg-slate-700 flex items-center justify-center">
          <File size={24} className="text-slate-500" />
        </div>
      );
    }

    // Xây dựng URL cho thumbnail
    const thumbnailUrl = `${API_BASE_URL}/frame/${video.thumbnail.split(/[/\\]/).pop()}`;
    return (
      <img
        src={thumbnailUrl}
        alt={video.title}
        className="w-full h-full object-cover"
        loading="lazy"
        crossOrigin="anonymous"
      />
    );
  };

  return (
    <div>
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-medium">Your Videos</h2>
        <button 
          onClick={handleUploadClick}
          className="text-xs text-teal-400 hover:text-teal-300 flex items-center gap-1"
        >
          <Upload size={12} />
          Add New Video
        </button>
      </div>

      {isLoading ? (
        <div className="flex justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-teal-500"></div>
        </div>
      ) : videos.length === 0 ? (
        <div className="text-center py-8">
          <div className="w-12 h-12 rounded-full bg-slate-700 flex items-center justify-center mx-auto mb-3">
            <Upload size={20} className="text-slate-500" />
          </div>
          <p className="text-sm text-slate-400">No videos in your library</p>
          <p className="text-xs text-slate-500 mt-1">Upload videos to begin</p>
        </div>
      ) : (
        <div className="space-y-3">
          {videos.map((video) => (
            <div
              key={video.id}
              onClick={() => handleSelectVideo(video)}
              className={`group cursor-pointer transition-all duration-200 hover:bg-slate-750 ${
                currentVideo && currentVideo.id === video.id ? 'bg-slate-700' : 'bg-slate-800'
              } rounded-lg overflow-hidden`}
            >
              <div className="flex">
                <div className="w-24 h-16 flex-shrink-0 relative overflow-hidden">
                  {renderThumbnail(video)}
                  <div className="absolute bottom-1 right-1 bg-black/70 text-white text-xs px-1 rounded">
                    {formatDuration(video.duration)}
                  </div>
                </div>
                <div className="p-3 flex-1 min-w-0">
                  <h3 className="font-medium text-sm mb-1 truncate group-hover:text-teal-400 transition-colors duration-200">
                    {video.title}
                  </h3>
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                    <div className="flex items-center text-xs text-slate-400">
                      <Calendar size={10} className="mr-1" />
                      <span>{video.uploadDate}</span>
                    </div>
                    <span className="text-xs text-slate-500">•</span>
                    <span className="text-xs text-slate-400">{video.resolution}</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Add New Video button at bottom */}
      <button
        onClick={handleUploadClick}
        className="w-full p-2 mt-4 border border-dashed border-slate-600 rounded-lg text-slate-400 hover:text-teal-500 hover:border-teal-500 transition-colors duration-200 flex items-center justify-center gap-2"
      >
        <Film size={16} />
        <span>Add New Video</span>
      </button>

      {/* Upload Modal */}
      {showUpload && (
        <div className="fixed inset-0 flex items-center justify-center bg-black/70 z-50">
          <div className="bg-slate-800 rounded-lg p-6 w-full max-w-md">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-medium">Upload New Video</h3>
              <button 
                onClick={handleCloseUpload}
                className="text-slate-400 hover:text-white"
                disabled={isUploading}
              >
                <X size={20} />
              </button>
            </div>
            
            {isUploading ? (
              <div className="text-center py-6">
                <div className="mb-4 mx-auto w-16 h-16 rounded-full bg-slate-700 flex items-center justify-center">
                  <Loader size={24} className="text-teal-500 animate-spin" />
                </div>
                <h4 className="font-medium mb-2">Processing Video...</h4>
                <p className="text-sm text-slate-400 mb-4">
                  This may take a few minutes
                </p>
                <div className="w-full h-2 bg-slate-700 rounded-full mb-2">
                  <div 
                    className="h-full bg-teal-500 rounded-full transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  ></div>
                </div>
                <p className="text-xs text-slate-500">{uploadProgress}% complete</p>
              </div>
            ) : (
              <div className="text-center py-6">
                <div className="mb-4 mx-auto w-16 h-16 rounded-full bg-slate-700 flex items-center justify-center">
                  <Upload size={24} className="text-slate-400" />
                </div>
                <h4 className="font-medium mb-2">Select Video File</h4>
                <p className="text-sm text-slate-400 mb-4">
                  Supported formats: MP4, AVI, MOV
                </p>
                <input
                  type="file"
                  accept="video/*"
                  onChange={handleFileChange}
                  className="hidden"
                  ref={fileInputRef}
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="px-4 py-2 bg-teal-500 text-white rounded-md hover:bg-teal-400 transition-colors duration-200"
                >
                  Browse Files
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};