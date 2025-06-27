import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Loader2, Settings, RefreshCw, Download, Search, Zap } from 'lucide-react';
import { useVideo } from '../context/VideoContext';

const API_URL = 'http://localhost:5000/api';

interface VideoInfo {
  name: string;
  embeddings_file: string;
  video_path: string;
}

interface UMAPParameters {
  n_neighbors: number;
  min_dist: number;
  metric: string;
}

interface FrameMetadata {
  video_name: string;
  frameidx: number;
  filepath: string;
  frame_id: number;
  text?: string;
  text_confidence?: number;
  object?: string;
  object_confidence?: number;
}

interface UMAPPoint {
  x: number;
  y: number;
  videoLabel: string;
  frameIndex: number;
  metadata: FrameMetadata;
}

export const VisualizationPanel = () => {
  const { currentVideo } = useVideo();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [availableVideos, setAvailableVideos] = useState<VideoInfo[]>([]);
  const [selectedVideos, setSelectedVideos] = useState<string[]>([]);
  const [points, setPoints] = useState<UMAPPoint[]>([]);
  const [showSettings, setShowSettings] = useState(false);
  const [showInfo, setShowInfo] = useState<UMAPPoint | null>(null);
  const [uMapParams, setUMapParams] = useState<UMAPParameters>({
    n_neighbors: 15,
    min_dist: 0.1,
    metric: 'cosine'
  });
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // Fetch available videos on mount
  useEffect(() => {
    fetchAvailableVideos();
  }, []);
  
  // Update canvas when points change
  useEffect(() => {
    if (points.length > 0) {
      drawVisualization();
    }
  }, [points]);
  
  // Select current video when it changes
  useEffect(() => {
    if (currentVideo && availableVideos.length > 0) {
      // Extract video name from current video (might need adjustment based on your data structure)
      const videoName = currentVideo.title;
      if (!selectedVideos.includes(videoName) && availableVideos.some(v => v.name === videoName)) {
        setSelectedVideos([...selectedVideos, videoName]);
      }
    }
  }, [currentVideo, availableVideos]);

  const fetchAvailableVideos = async () => {
    try {
      const response = await axios.get(`${API_URL}/videos/available`);
      setAvailableVideos(response.data.available_videos || []);
    } catch (error) {
      console.error('Error fetching available videos:', error);
      setError('Failed to fetch available videos');
    }
  };

  const generateUMAP = async () => {
    if (selectedVideos.length === 0 && availableVideos.length > 0) {
      // If no videos selected, use all available videos
      setSelectedVideos(availableVideos.map(video => video.name));
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await axios.post(`${API_URL}/visualization/umap`, {
        video_names: selectedVideos.length > 0 ? selectedVideos : null, // null means all videos
        n_neighbors: uMapParams.n_neighbors,
        min_dist: uMapParams.min_dist,
        metric: uMapParams.metric
      });
      
      // Process the response data
      const { coordinates, video_labels, metadata } = response.data;
      
      // Create points array
      const processedPoints = coordinates.map((coord: number[], i: number) => ({
        x: coord[0],
        y: coord[1],
        videoLabel: video_labels[i],
        frameIndex: metadata[i].frameidx,
        metadata: metadata[i]
      }));
      
      setPoints(processedPoints);
    } catch (error) {
      console.error('Error generating UMAP:', error);
      setError('Failed to generate UMAP visualization');
    } finally {
      setIsLoading(false);
    }
  };

  const drawVisualization = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Find min and max to normalize coordinates
    const xValues = points.map(p => p.x);
    const yValues = points.map(p => p.y);
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);
    
    // Add some padding
    const padding = 20;
    const width = canvas.width - padding * 2;
    const height = canvas.height - padding * 2;
    
    // Generate colors for different videos
    const uniqueVideos = [...new Set(points.map(p => p.videoLabel))];
    const colorMap = Object.fromEntries(
      uniqueVideos.map((video, index) => [
        video,
        `hsl(${(index * 360) / uniqueVideos.length}, 80%, 60%)`
      ])
    );
    
    // Draw points
    points.forEach(point => {
      // Normalize coordinates to fit canvas
      const x = padding + ((point.x - xMin) / (xMax - xMin)) * width;
      const y = padding + ((point.y - yMin) / (yMax - yMin)) * height;
      
      // Draw point
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fillStyle = colorMap[point.videoLabel] || '#888';
      ctx.fill();
    });
    
    // Draw legend
    const legendX = 10;
    let legendY = 20;
    
    ctx.font = '12px Arial';
    ctx.textAlign = 'left';
    uniqueVideos.forEach(video => {
      // Draw color box
      ctx.fillStyle = colorMap[video] || '#888';
      ctx.fillRect(legendX, legendY - 10, 10, 10);
      
      // Draw label
      ctx.fillStyle = '#fff';
      ctx.fillText(video, legendX + 15, legendY);
      
      legendY += 15;
    });
  };

  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || points.length === 0) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Find min and max to normalize coordinates
    const xValues = points.map(p => p.x);
    const yValues = points.map(p => p.y);
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);
    
    // Add some padding
    const padding = 20;
    const width = canvas.width - padding * 2;
    const height = canvas.height - padding * 2;
    
    // Find the nearest point
    let minDist = Infinity;
    let nearestPoint = null;
    
    points.forEach(point => {
      // Convert UMAP coordinates to canvas coordinates
      const px = padding + ((point.x - xMin) / (xMax - xMin)) * width;
      const py = padding + ((point.y - yMin) / (yMax - yMin)) * height;
      
      // Calculate distance
      const dist = Math.sqrt((x - px) ** 2 + (y - py) ** 2);
      if (dist < minDist && dist < 10) {  // Only consider points within 10 pixels
        minDist = dist;
        nearestPoint = point;
      }
    });
    
    setShowInfo(nearestPoint);
  };

  const handleToggleVideo = (videoName: string) => {
    if (selectedVideos.includes(videoName)) {
      setSelectedVideos(selectedVideos.filter(v => v !== videoName));
    } else {
      setSelectedVideos([...selectedVideos, videoName]);
    }
  };

  const handleSelectAll = () => {
    setSelectedVideos(availableVideos.map(v => v.name));
  };

  const handleDeselectAll = () => {
    setSelectedVideos([]);
  };

  const downloadVisualization = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const link = document.createElement('a');
    link.download = 'umap_visualization.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  return (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b border-slate-700">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Zap size={20} className="text-teal-500" />
            <h2 className="text-lg font-medium">UMAP Visualization</h2>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 rounded-lg border border-slate-600 text-slate-300 hover:bg-slate-700"
              title="Settings"
            >
              <Settings size={16} />
            </button>
            <button
              onClick={generateUMAP}
              className="p-2 rounded-lg border border-slate-600 text-slate-300 hover:bg-slate-700"
              title="Refresh"
              disabled={isLoading}
            >
              {isLoading ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                <RefreshCw size={16} />
              )}
            </button>
            <button
              onClick={downloadVisualization}
              className="p-2 rounded-lg border border-slate-600 text-slate-300 hover:bg-slate-700"
              title="Download"
              disabled={points.length === 0}
            >
              <Download size={16} />
            </button>
          </div>
        </div>

        {showSettings && (
          <div className="mb-4 p-4 bg-slate-800 border border-slate-700 rounded-lg">
            <h3 className="font-medium mb-3">UMAP Settings</h3>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">
                  n_neighbors: {uMapParams.n_neighbors}
                </label>
                <input
                  type="range"
                  min="2"
                  max="100"
                  value={uMapParams.n_neighbors}
                  onChange={(e) =>
                    setUMapParams({ ...uMapParams, n_neighbors: parseInt(e.target.value) })
                  }
                  className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-teal-500"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-1">
                  min_dist: {uMapParams.min_dist}
                </label>
                <input
                  type="range"
                  min="0.01"
                  max="1"
                  step="0.01"
                  value={uMapParams.min_dist}
                  onChange={(e) =>
                    setUMapParams({ ...uMapParams, min_dist: parseFloat(e.target.value) })
                  }
                  className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-teal-500"
                />
              </div>
              
              <div className="col-span-2">
                <label className="block text-sm font-medium mb-1">Metric</label>
                <select
                  value={uMapParams.metric}
                  onChange={(e) =>
                    setUMapParams({ ...uMapParams, metric: e.target.value })
                  }
                  className="w-full p-2 bg-slate-700 border border-slate-600 rounded text-sm"
                >
                  <option value="cosine">Cosine</option>
                  <option value="euclidean">Euclidean</option>
                  <option value="manhattan">Manhattan</option>
                </select>
              </div>
            </div>
            
            <div className="mt-4">
              <div className="flex justify-between mb-2">
                <h4 className="text-sm font-medium">Select Videos</h4>
                <div className="flex gap-2">
                  <button
                    onClick={handleSelectAll}
                    className="text-xs text-teal-400 hover:text-teal-300"
                  >
                    Select All
                  </button>
                  <button
                    onClick={handleDeselectAll}
                    className="text-xs text-red-400 hover:text-red-300"
                  >
                    Deselect All
                  </button>
                </div>
              </div>
              
              <div className="max-h-32 overflow-y-auto p-2 bg-slate-700 rounded border border-slate-600">
                {availableVideos.length === 0 ? (
                  <p className="text-sm text-slate-400">No videos available</p>
                ) : (
                  <div className="space-y-1">
                    {availableVideos.map((video) => (
                      <div
                        key={video.name}
                        className="flex items-center gap-2"
                      >
                        <input
                          type="checkbox"
                          id={`video-${video.name}`}
                          checked={selectedVideos.includes(video.name)}
                          onChange={() => handleToggleVideo(video.name)}
                          className="h-4 w-4 rounded border-gray-300 text-teal-600 focus:ring-teal-500"
                        />
                        <label
                          htmlFor={`video-${video.name}`}
                          className="text-sm text-slate-300 cursor-pointer flex-1 truncate"
                          title={video.name}
                        >
                          {video.name}
                        </label>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="p-3 bg-red-500/20 border border-red-500 rounded-lg text-sm">
            {error}
          </div>
        )}
      </div>

      <div className="flex-1 relative overflow-hidden p-4">
        {isLoading && (
          <div className="absolute inset-0 bg-slate-900/70 flex items-center justify-center z-10">
            <div className="flex flex-col items-center">
              <Loader2 size={32} className="animate-spin text-teal-500 mb-2" />
              <p className="text-teal-400 text-sm">Generating UMAP...</p>
            </div>
          </div>
        )}

        {points.length === 0 && !isLoading ? (
          <div className="h-full flex flex-col items-center justify-center text-slate-500">
            <Search size={32} className="mb-2" />
            <p>No visualization data</p>
            <button
              onClick={generateUMAP}
              className="mt-4 px-4 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-500 transition-colors"
            >
              Generate Visualization
            </button>
          </div>
        ) : (
          <canvas
            ref={canvasRef}
            onClick={handleCanvasClick}
            width="800"
            height="600"
            className="w-full h-full bg-slate-800 border border-slate-700 rounded-lg cursor-crosshair"
          />
        )}
      </div>

      {showInfo && (
        <div className="absolute bottom-4 right-4 p-4 bg-slate-800 border border-slate-700 rounded-lg shadow-lg max-w-md">
          <div className="flex justify-between items-start">
            <h3 className="font-medium">Point Details</h3>
            <button
              onClick={() => setShowInfo(null)}
              className="text-slate-400 hover:text-slate-300"
            >
              <X size={16} />
            </button>
          </div>
          
          <div className="mt-2 space-y-2">
            <div className="flex gap-2">
              <span className="text-slate-400">Video:</span>
              <span>{showInfo.videoLabel}</span>
            </div>
            
            <div className="flex gap-2">
              <span className="text-slate-400">Frame:</span>
              <span>{showInfo.frameIndex}</span>
            </div>
            
            {showInfo.metadata.text && (
              <div className="flex gap-2">
                <span className="text-slate-400">Text:</span>
                <span>{showInfo.metadata.text}</span>
              </div>
            )}
            
            {showInfo.metadata.object && (
              <div className="flex gap-2">
                <span className="text-slate-400">Object:</span>
                <span>{showInfo.metadata.object}</span>
              </div>
            )}
            
            <div className="mt-2">
              <img
                src={showInfo.metadata.filepath}
                alt="Frame"
                className="max-h-32 rounded border border-slate-600"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Fix TypeScript error by declaring X component
const X = ({ size }: { size: number }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="18" y1="6" x2="6" y2="18"></line>
    <line x1="6" y1="6" x2="18" y2="18"></line>
  </svg>
); 