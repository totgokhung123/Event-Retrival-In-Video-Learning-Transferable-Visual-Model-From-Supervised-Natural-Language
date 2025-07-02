import React, { useState, useRef } from 'react';
import { File, Upload, Calendar, Film, X, Loader } from 'lucide-react';
import { useVideo } from '../context/VideoContext';
import { formatDuration } from '../utils/formatters';
import { VideoData } from '../types';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

export const VideoLibrary = () => {
  const { videos, setCurrentVideo, currentVideo, isLoading, loadVideoEvents, uploadVideo } = useVideo();
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [showUpload, setShowUpload] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState(false);

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
  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setSelectedFile(file);
    setError(null);
    setUploadProgress(0);
  };

  // Trong hàm handleUpload, thêm phần để lấy mô hình đã chọn và gửi lên server
  const handleUpload = async () => {
    if (!selectedFile) {
      setError("Please select a video file");
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('video', selectedFile);
      
      // Thêm model vào formData
      const activeModelResponse = await axios.get(`${API_BASE_URL}/models/active`);
      const activeModel = activeModelResponse.data?.active_model || 'original';
      formData.append('model', activeModel);
      
      const config = {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        onUploadProgress: (progressEvent: any) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(percentCompleted);
        }
      };

      const response = await axios.post(`${API_BASE_URL}/upload-video`, formData, config);
      
      if (response.data && response.data.status === 'success') {
        setIsUploading(false);
        setUploadSuccess(true);
        setSelectedFile(null);
        
        // Add new video to videos list
        if (response.data.video) {
          uploadVideo(response.data.video);
        }
        
        // Show notification for a few seconds
        setTimeout(() => {
          setUploadSuccess(false);
          setShowUpload(false);
        }, 3000);
      }
    } catch (err: any) {
      setIsUploading(false);
      setError(err.response?.data?.error || "Error uploading video. Please try again.");
      console.error("Error uploading video:", err);
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
                  onChange={handleFileSelect}
                  className="hidden"
                  ref={fileInputRef}
                />
                
                {selectedFile ? (
                  <div className="mb-4">
                    <div className="p-3 bg-slate-700 rounded-md flex items-center">
                      <Film size={16} className="text-teal-400 mr-2" />
                      <span className="text-sm truncate">{selectedFile.name}</span>
                    </div>
                  </div>
                ) : null}
                
                <div className="flex justify-center gap-3">
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="px-4 py-2 border border-slate-600 text-slate-300 rounded-md hover:bg-slate-700 transition-colors duration-200"
                  >
                    Browse Files
                  </button>
                  
                  {selectedFile && (
                    <button
                      onClick={handleUpload}
                      className="px-4 py-2 bg-teal-500 text-white rounded-md hover:bg-teal-400 transition-colors duration-200"
                    >
                      Upload
                    </button>
                  )}
                </div>
                
                {error && (
                  <div className="mt-4 p-2 bg-red-500/20 border border-red-500/30 rounded-md">
                    <p className="text-sm text-red-300">{error}</p>
                  </div>
                )}
                
                {uploadSuccess && (
                  <div className="mt-4 p-2 bg-green-500/20 border border-green-500/30 rounded-md">
                    <p className="text-sm text-green-300">Video uploaded successfully!</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};