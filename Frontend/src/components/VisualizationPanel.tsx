import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Loader2, Settings, RefreshCw, Download, Search, Zap, ZoomIn, ZoomOut, Maximize, Move, X } from 'lucide-react';
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
  original_filepath?: string; // Thêm trường này để lưu đường dẫn gốc
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
  const [selectedPoints, setSelectedPoints] = useState<UMAPPoint[]>([]);
  const [uMapParams, setUMapParams] = useState<UMAPParameters>({
    n_neighbors: 15,
    min_dist: 0.1,
    metric: 'cosine'
  });
  
  // Canvas view state
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [selectionBox, setSelectionBox] = useState<{
    start: {x: number, y: number}, 
    end: {x: number, y: number},
    isCtrlKey?: boolean
  } | null>(null);
  const [isSelecting, setIsSelecting] = useState(false);
  const [currentTool, setCurrentTool] = useState<'select' | 'pan'>('select');
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Lưu trữ các thông số về không gian dữ liệu, không dùng state
  const dataSpaceRef = useRef<{xMin: number, xMax: number, yMin: number, yMax: number} | null>(null);
  
  // Fetch available videos on mount
  useEffect(() => {
    fetchAvailableVideos();
  }, []);
  
  // Update canvas when points or view state changes
  useEffect(() => {
    if (points.length > 0) {
      drawVisualization();
    }
  }, [points, scale, offset, selectionBox, selectedPoints]);
  
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

  // Helper function to convert từ tọa độ dữ liệu sang tọa độ màn hình
  const toScreenCoords = (point: {x: number, y: number}, dataSpace: {xMin: number, xMax: number, yMin: number, yMax: number}, canvas: HTMLCanvasElement, padding: number = 20) => {
    // Chuẩn hóa tọa độ trong không gian dữ liệu (0-1)
    const normalizedX = (point.x - dataSpace.xMin) / (dataSpace.xMax - dataSpace.xMin);
    const normalizedY = (point.y - dataSpace.yMin) / (dataSpace.yMax - dataSpace.yMin);
    
    // Chuyển sang tọa độ canvas
    const width = canvas.width - padding * 2;
    const height = canvas.height - padding * 2;
    const canvasX = padding + normalizedX * width;
    const canvasY = padding + normalizedY * height;
    
    // Áp dụng scale và offset
    return {
      x: offset.x + canvasX * scale,
      y: offset.y + canvasY * scale
    };
  };

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
      setSelectedPoints([]);
      resetViewport();
    } catch (error) {
      console.error('Error generating UMAP:', error);
      setError('Failed to generate UMAP visualization');
    } finally {
      setIsLoading(false);
    }
  };

  const resetViewport = () => {
    // Base scale on number of points to avoid overcrowding for large datasets
    const pointCount = points.length;
    const autoScale = pointCount > 1000 ? 0.8 : pointCount > 500 ? 0.9 : 1;
    setScale(autoScale);
    setOffset({ x: 0, y: 0 });
    
    // Ensure canvas dimensions are properly set
    if (canvasRef.current && containerRef.current) {
      const container = containerRef.current;
      canvasRef.current.width = container.clientWidth;
      canvasRef.current.height = container.clientHeight;
    }
    
    // Reset selection
    setSelectionBox(null);
    setIsSelecting(false);
    setIsDragging(false);
  };
  
  const drawVisualization = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#1e293b'; // bg-slate-800
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Find min and max to normalize coordinates
    if (points.length === 0) return;
    
    const xValues = points.map(p => p.x);
    const yValues = points.map(p => p.y);
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);
    
    // Cập nhật dataSpace ref - thay vì dùng state normalizedCoords
    dataSpaceRef.current = { xMin, xMax, yMin, yMax };
    
    // Add some padding
    const padding = 20;
    
    // Generate colors for different videos
    const uniqueVideos = [...new Set(points.map(p => p.videoLabel))];
    const colorMap = Object.fromEntries(
      uniqueVideos.map((video, index) => [
        video,
        `hsl(${(index * 360) / uniqueVideos.length}, 80%, 60%)`
      ])
    );
    
    // Draw points
    ctx.save();
    
    // Adjust point size based on canvas size and scale
    const pointRadius = Math.max(2, Math.min(3, 3 / Math.sqrt(scale)));
    
    points.forEach(point => {
      // Sử dụng helper function thay vì tính toán trực tiếp
      const { x: screenX, y: screenY } = toScreenCoords(
        point,
        { xMin, xMax, yMin, yMax },
        canvas,
        padding
      );
      
      // Draw point
      ctx.beginPath();
      ctx.arc(screenX, screenY, pointRadius, 0, Math.PI * 2);
      
      // Helper function to check if a point is selected
      const isPointSelected = (p: UMAPPoint) => selectedPoints.some(
        selected => selected.metadata.frame_id === p.metadata.frame_id && 
                   selected.videoLabel === p.videoLabel
      );
      
      // Highlight selected points
      if (isPointSelected(point)) {
        ctx.fillStyle = '#10b981'; // text-emerald-500
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
        ctx.fill();
        ctx.stroke();
      } else {
        ctx.fillStyle = colorMap[point.videoLabel] || '#888';
        ctx.fill();
      }
    });
    
    ctx.restore();
    
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
    
    // Draw selection box if active
    if (selectionBox) {
      ctx.strokeStyle = '#10b981'; // text-emerald-500
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      
      const width = selectionBox.end.x - selectionBox.start.x;
      const height = selectionBox.end.y - selectionBox.start.y;
      
      ctx.strokeRect(selectionBox.start.x, selectionBox.start.y, width, height);
      ctx.setLineDash([]);
      
      // Semi-transparent fill
      ctx.fillStyle = 'rgba(16, 185, 129, 0.1)'; // text-emerald-500 with opacity
      ctx.fillRect(selectionBox.start.x, selectionBox.start.y, width, height);
    }
    
    // Draw tool indicators
    ctx.fillStyle = '#fff';
    ctx.font = '12px Arial';
    ctx.textAlign = 'left';
    ctx.fillText(`Tool: ${currentTool === 'select' ? 'Selection' : 'Pan'}`, 10, canvas.height - 10);
    ctx.fillText(`Zoom: ${Math.round(scale * 100)}%`, 10, canvas.height - 30);
  };

  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (isSelecting || isDragging || currentTool === 'pan') return;
    
    const canvas = canvasRef.current;
    if (!canvas || points.length === 0 || !dataSpaceRef.current) return;
    
    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;
    
    // Find the nearest point
    let minDist = Infinity;
    let nearestPoint: UMAPPoint | null = null;
    
    // Use the dataSpaceRef instead of normalizedCoords
    const dataSpace = dataSpaceRef.current;
    const padding = 20;
    
    points.forEach(point => {
      // Sử dụng helper function để tính tọa độ màn hình của điểm
      const screenPos = toScreenCoords(point, dataSpace, canvas, padding);
      
      // Calculate distance to mouse position
      const dist = Math.sqrt(
        (mouseX - screenPos.x) ** 2 + 
        (mouseY - screenPos.y) ** 2
      );
      
      // Adjust selection radius based on scale - smaller radius when zoomed out
      const selectionRadius = 10 / Math.sqrt(scale);
      
      if (dist < minDist && dist < selectionRadius) {
        minDist = dist;
        nearestPoint = point;
      }
    });
    
    // Helper function to check if two points are the same (by metadata.frame_id and videoLabel)
    const isSamePoint = (p1: UMAPPoint, p2: UMAPPoint) => 
      p1.metadata.frame_id === p2.metadata.frame_id && p1.videoLabel === p2.videoLabel;
    
    if (nearestPoint) {
      // If holding Ctrl/Cmd key, add to selection or remove if already selected
      if (event.ctrlKey || event.metaKey) {
        // Check if point is already selected
        const existingPointIndex = selectedPoints.findIndex(p => isSamePoint(p, nearestPoint!));
        
        if (existingPointIndex >= 0) {
          // Remove the point if already selected
          const newSelection = [...selectedPoints];
          newSelection.splice(existingPointIndex, 1);
          setSelectedPoints(newSelection);
        } else {
          // Add the point to selection
          setSelectedPoints([...selectedPoints, nearestPoint]);
        }
      } else {
        // Replace selection
        setSelectedPoints([nearestPoint]);
      }
    } else if (!(event.ctrlKey || event.metaKey)) {
      // Clear selection if clicking empty space without Ctrl/Cmd
      setSelectedPoints([]);
    }
  };
  
  const handleCanvasMouseDown = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (currentTool === 'select') {
      // Start selection box
      setIsSelecting(true);
      setSelectionBox({
        start: { x: event.nativeEvent.offsetX, y: event.nativeEvent.offsetY },
        end: { x: event.nativeEvent.offsetX, y: event.nativeEvent.offsetY },
        isCtrlKey: event.ctrlKey || event.metaKey
      });
    } else {
      // Start panning
      setIsDragging(true);
      setDragStart({ x: event.clientX - offset.x, y: event.clientY - offset.y });
    }
  };
  
  const handleCanvasMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (isSelecting && selectionBox) {
      // Update selection box, preserving isCtrlKey
      const updatedSelectionBox = {
        start: selectionBox.start,
        end: { x: event.nativeEvent.offsetX, y: event.nativeEvent.offsetY },
        isCtrlKey: selectionBox.isCtrlKey || event.ctrlKey || event.metaKey
      };
      
      // Only update if there's a change
      if (
        updatedSelectionBox.end.x !== selectionBox.end.x || 
        updatedSelectionBox.end.y !== selectionBox.end.y ||
        updatedSelectionBox.isCtrlKey !== selectionBox.isCtrlKey
      ) {
        setSelectionBox(updatedSelectionBox);
      }
    } else if (isDragging) {
      // Update offset for panning
      setOffset({
        x: event.clientX - dragStart.x,
        y: event.clientY - dragStart.y
      });
    }
  };
  
  const handleCanvasMouseUp = (event: React.MouseEvent<HTMLCanvasElement>) => {
    // Lấy tọa độ trực tiếp từ event thay vì dùng selectionBox.end
    const currentX = event.nativeEvent.offsetX;
    const currentY = event.nativeEvent.offsetY;
    
    if (isSelecting && selectionBox) {
      const canvas = canvasRef.current;
      if (!canvas || !dataSpaceRef.current) {
        setIsSelecting(false);
        setSelectionBox(null);
        return;
      }
      
      // Trực tiếp tính tọa độ selection box từ event và selectionBox.start
      const minX = Math.min(selectionBox.start.x, currentX);
      const maxX = Math.max(selectionBox.start.x, currentX);
      const minY = Math.min(selectionBox.start.y, currentY);
      const maxY = Math.max(selectionBox.start.y, currentY);
      
      // Tăng threshold lên để xử lý selection nhỏ tốt hơn
      const isSmallSelection = 
        Math.abs(currentX - selectionBox.start.x) < 10 && 
        Math.abs(currentY - selectionBox.start.y) < 10;
        
      if (isSmallSelection) {
        console.log("Selection too small, treating as click");
        // Treat as a click - already handled by handleCanvasClick
        setIsSelecting(false);
        setSelectionBox(null);
        return;
      }
      
      // Sử dụng dataSpaceRef thay vì state normalizedCoords
      const dataSpace = dataSpaceRef.current;
      const padding = 20;
      
      // Tạo mảng lưu các điểm được chọn
      const selectedPointsInBox: UMAPPoint[] = [];
      
      // Kiểm tra từng điểm xem có nằm trong selection box không
      for (const point of points) {
        // Sử dụng helper function để tính tọa độ màn hình
        const screenPos = toScreenCoords(point, dataSpace, canvas, padding);
        
        // Kiểm tra nếu điểm nằm trong selection box
        if (
          screenPos.x >= minX && 
          screenPos.x <= maxX && 
          screenPos.y >= minY && 
          screenPos.y <= maxY
        ) {
          selectedPointsInBox.push(point);
        }
      }
      
      // Nếu không tìm thấy điểm nào, giữ nguyên trạng thái
      if (selectedPointsInBox.length === 0) {
        console.log("No points found in selection box");
        setIsSelecting(false);
        setSelectionBox(null);
        return;
      }
      
      // Loại bỏ các điểm trùng lặp dựa trên frame_id và videoLabel
      const uniquePointsMap = new Map<string, UMAPPoint>();
      
      for (const point of selectedPointsInBox) {
        const pointKey = `${point.videoLabel}-${point.metadata.frame_id}`;
        if (!uniquePointsMap.has(pointKey)) {
          uniquePointsMap.set(pointKey, point);
        }
      }
      
      // Chuyển đổi map thành mảng
      const uniqueSelectedPoints = Array.from(uniquePointsMap.values());
      
      // Cập nhật selection dựa trên trạng thái phím Ctrl
      const isCtrlKey = selectionBox.isCtrlKey || event.ctrlKey || event.metaKey;
      if (isCtrlKey) {
        // Thêm vào selection hiện tại, tránh trùng lặp
        const newSelection = [...selectedPoints];
        const existingKeys = new Set(newSelection.map(p => `${p.videoLabel}-${p.metadata.frame_id}`));
        
        for (const point of uniqueSelectedPoints) {
          const pointKey = `${point.videoLabel}-${point.metadata.frame_id}`;
          if (!existingKeys.has(pointKey)) {
            newSelection.push(point);
          }
        }
        
        setSelectedPoints(newSelection);
      } else {
        // Thay thế selection
        setSelectedPoints(uniqueSelectedPoints);
      }
    }
    
    // Clean up selection state
    setIsSelecting(false);
    setSelectionBox(null);
    setIsDragging(false);
  };
  
  const handleCanvasMouseLeave = () => {
    setIsDragging(false);
    setIsSelecting(false);
  };

  const handleWheel = (event: React.WheelEvent) => {
    event.preventDefault();
    
    const zoomFactor = event.deltaY > 0 ? 0.9 : 1.1; // Zoom out/in
    const newScale = Math.max(0.1, Math.min(5, scale * zoomFactor));
    
    // Zoom centered at mouse position
    const rect = canvasRef.current!.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;
    
    const oldMousePosX = (mouseX - offset.x) / scale;
    const oldMousePosY = (mouseY - offset.y) / scale;
    
    const newOffset = {
      x: mouseX - oldMousePosX * newScale,
      y: mouseY - oldMousePosY * newScale
    };
    
    setScale(newScale);
    setOffset(newOffset);
  };

  const handleZoomIn = () => {
    const newScale = Math.min(5, scale * 1.2);
    setScale(newScale);
  };
  
  const handleZoomOut = () => {
    const newScale = Math.max(0.1, scale * 0.8);
    setScale(newScale);
  };
  
  const handleResetZoom = () => {
    resetViewport();
  };
  
  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
    
    // Allow the resize to take effect before recalculating
    setTimeout(() => {
      if (canvasRef.current && containerRef.current) {
        const container = containerRef.current;
        canvasRef.current.width = container.clientWidth;
        canvasRef.current.height = container.clientHeight;
        drawVisualization();
      }
    }, 100);
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
  
  const toggleTool = () => {
    setCurrentTool(currentTool === 'select' ? 'pan' : 'select');
  };

  // Make canvas responsive
  useEffect(() => {
    const updateCanvasSize = () => {
      if (canvasRef.current && containerRef.current) {
        const container = containerRef.current;
        canvasRef.current.width = container.clientWidth;
        canvasRef.current.height = container.clientHeight;
        drawVisualization();
      }
    };
    
    window.addEventListener('resize', updateCanvasSize);
    updateCanvasSize(); // Initial size
    
    return () => window.removeEventListener('resize', updateCanvasSize);
  }, []);

  return (
    <div 
        className={`h-full flex ${isFullscreen ? 'fixed inset-0 z-50 bg-slate-900' : ''}`}
        role="application"
        aria-label="UMAP Visualization Dashboard"
      >
        {/* Left panel: Selected Events */}
        <section 
          className={`${isFullscreen ? 'w-1/4' : 'w-1/3'} border-r border-slate-700 flex flex-col`}
          aria-label="Selected Events Panel"
        >
          <header className="p-4 border-b border-slate-700">
            <h2 className="text-lg font-medium flex items-center">
              <Zap size={20} className="text-teal-500 mr-2" />
              Selected Events ({selectedPoints.length})
            </h2>
          </header>
          
          <div className="flex-1 overflow-y-auto p-2">
            {selectedPoints.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-slate-500">
                <Search size={32} className="mb-2" />
                <p>No events selected</p>
                <p className="text-sm text-slate-600 mt-2">Click or drag to select points on the visualization</p>
              </div>
            ) : (
              <ul className="space-y-4" role="list" aria-label="Selected events list">
                {selectedPoints.map((point, index) => (
                  <li 
                    key={index} 
                    className="p-3 bg-slate-800 rounded-lg border border-slate-700"
                    role="listitem"
                  >
                    <div className="flex justify-between">
                      <span className="font-medium">{point.videoLabel}</span>
                      <span className="text-slate-400">Frame: {point.frameIndex}</span>
                    </div>
                    
                    {point.metadata.text && (
                      <div className="mt-1 text-sm">
                        <span className="text-slate-400">Text:</span> {point.metadata.text}
                      </div>
                    )}
                    
                    {point.metadata.object && (
                      <div className="mt-1 text-sm">
                        <span className="text-slate-400">Object:</span> {point.metadata.object}
                      </div>
                    )}
                    
                    <div className="mt-2">
                      {point.metadata.filepath && point.metadata.filepath.trim() !== '' ? (
                        <div>
                          {/* Log để debug đường dẫn hình ảnh - sử dụng useEffect thay vì console.log trực tiếp */}
                          <img
                            src={point.metadata.filepath}
                            alt={`Frame ${point.frameIndex} from ${point.videoLabel} showing ${point.metadata.object || point.metadata.text || 'video frame'}`}
                            className="w-full h-auto rounded border border-slate-600"
                            loading="lazy"
                            onLoad={() => console.log("Image loaded successfully:", point.metadata.filepath)}
                            onError={(e) => {
                              console.error("Failed to load image:", point.metadata.filepath);
                              console.error("Original filepath:", point.metadata.original_filepath);
                              (e.target as HTMLImageElement).src = 'https://via.placeholder.com/400x300?text=Image+Not+Found';
                            }}
                          />
                        </div>
                      ) : (
                        <div className="w-full h-48 flex items-center justify-center bg-slate-700 rounded border border-slate-600">
                          <p className="text-slate-400">No image available</p>
                        </div>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>
        
        {/* Right panel: UMAP Visualization */}
        <section 
          className={`${isFullscreen ? 'w-3/4' : 'w-2/3'} flex flex-col`}
          aria-label="UMAP Visualization Panel"
        >
          <header className="p-4 border-b border-slate-700 flex flex-wrap gap-2 justify-between items-center">
            <div className="flex items-center">
              <h2 className="text-lg font-medium flex items-center">
                <Zap size={20} className="text-teal-500 mr-2" />
                UMAP Visualization
              </h2>
            </div>
            
            <div className="flex items-center gap-2 flex-wrap" role="toolbar" aria-label="Visualization controls">
              <div className="flex bg-slate-800 rounded-md" role="group" aria-label="Zoom and navigation controls">
                <button
                  onClick={handleZoomIn}
                  className="p-2 text-slate-300 hover:bg-slate-700 rounded-l-md"
                  title="Zoom In"
                  aria-label="Zoom In"
                >
                  <ZoomIn size={16} />
                </button>
                
                <button
                  onClick={handleZoomOut}
                  className="p-2 text-slate-300 hover:bg-slate-700"
                  title="Zoom Out"
                  aria-label="Zoom Out"
                >
                  <ZoomOut size={16} />
                </button>
                
                <button
                  onClick={handleResetZoom}
                  className="p-2 text-slate-300 hover:bg-slate-700"
                  title="Reset View"
                  aria-label="Reset View"
                >
                  <RefreshCw size={16} />
                </button>
                
                <button
                  onClick={toggleTool}
                  className={`p-2 text-slate-300 hover:bg-slate-700 ${currentTool === 'pan' ? 'bg-slate-600' : ''}`}
                  title="Toggle Tool (Select/Pan)"
                  aria-label={`Current tool: ${currentTool}. Click to switch to ${currentTool === 'select' ? 'Pan' : 'Select'} tool`}
                  aria-pressed={currentTool === 'pan'}
                >
                  {currentTool === 'select' ? <Search size={16} /> : <Move size={16} />}
                </button>
                
                <button
                  onClick={toggleFullscreen}
                  className="p-2 text-slate-300 hover:bg-slate-700 rounded-r-md"
                  title="Toggle Fullscreen"
                  aria-label="Toggle Fullscreen"
                  aria-pressed={isFullscreen}
                >
                  <Maximize size={16} />
                </button>
              </div>
              
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="p-2 rounded-lg border border-slate-600 text-slate-300 hover:bg-slate-700"
                title="Settings"
                aria-label="Settings"
                aria-expanded={showSettings}
              >
                <Settings size={16} />
              </button>
              
              <button
                onClick={generateUMAP}
                className="p-2 rounded-lg border border-slate-600 text-slate-300 hover:bg-slate-700"
                title="Regenerate UMAP"
                aria-label="Regenerate UMAP visualization"
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
                aria-label="Download visualization as image"
                disabled={points.length === 0}
              >
                <Download size={16} />
              </button>
            </div>
          </header>
          
          {showSettings && (
            <div 
              className="p-4 bg-slate-800 border-b border-slate-700" 
              role="region" 
              aria-label="Visualization Settings"
            >
              <div className="flex justify-between items-center mb-3">
                <h3 className="font-medium">UMAP Settings</h3>
                <div className="flex gap-2">
                  <button
                    onClick={handleSelectAll}
                    className="text-xs px-2 py-1 bg-teal-600 text-white rounded hover:bg-teal-500"
                  >
                    Select All Videos
                  </button>
                  <button
                    onClick={handleDeselectAll}
                    className="text-xs px-2 py-1 bg-slate-600 text-white rounded hover:bg-slate-500"
                  >
                    Deselect All
                  </button>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label htmlFor="n_neighbors" className="block text-sm font-medium mb-1">
                    n_neighbors: {uMapParams.n_neighbors}
                  </label>
                  <input
                    id="n_neighbors"
                    type="range"
                    min="2"
                    max="100"
                    value={uMapParams.n_neighbors}
                    onChange={(e) =>
                      setUMapParams({ ...uMapParams, n_neighbors: parseInt(e.target.value) })
                    }
                    className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-teal-500"
                    aria-valuemin={2}
                    aria-valuemax={100}
                    aria-valuenow={uMapParams.n_neighbors}
                  />
                </div>
                
                <div>
                  <label htmlFor="min_dist" className="block text-sm font-medium mb-1">
                    min_dist: {uMapParams.min_dist}
                  </label>
                  <input
                    id="min_dist"
                    type="range"
                    min="0.01"
                    max="1"
                    step="0.01"
                    value={uMapParams.min_dist}
                    onChange={(e) =>
                      setUMapParams({ ...uMapParams, min_dist: parseFloat(e.target.value) })
                    }
                    className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-teal-500"
                    aria-valuemin={0.01}
                    aria-valuemax={1}
                    aria-valuenow={uMapParams.min_dist}
                  />
                </div>
                
                <div>
                  <label htmlFor="metric" className="block text-sm font-medium mb-1">Metric</label>
                  <select
                    id="metric"
                    value={uMapParams.metric}
                    onChange={(e) =>
                      setUMapParams({ ...uMapParams, metric: e.target.value })
                    }
                    className="w-full p-2 bg-slate-700 border border-slate-600 rounded text-sm"
                    aria-label="Select distance metric for UMAP"
                  >
                    <option value="cosine">Cosine</option>
                    <option value="euclidean">Euclidean</option>
                    <option value="manhattan">Manhattan</option>
                  </select>
                </div>
              </div>
              
              <div className="mt-4">
                <h4 className="text-sm font-medium mb-2">Select Videos</h4>
                <div className="max-h-32 overflow-y-auto p-2 bg-slate-700 rounded border border-slate-600">
                  {availableVideos.length === 0 ? (
                    <p className="text-sm text-slate-400">No videos available</p>
                  ) : (
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2" role="group" aria-label="Available videos">
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
                            aria-label={`Select video: ${video.name}`}
                          />
                          <label
                            htmlFor={`video-${video.name}`}
                            className="text-sm text-slate-300 cursor-pointer truncate"
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
            <div 
              className="p-3 mx-4 mt-4 bg-red-500/20 border border-red-500 rounded-lg text-sm" 
              role="alert" 
              aria-live="assertive"
            >
              {error}
            </div>
          )}

          <div
            ref={containerRef}
            className="relative flex-1 overflow-hidden"
            onContextMenu={(e) => e.preventDefault()}
          >
            {isLoading && (
              <div 
                className="absolute inset-0 bg-slate-900/70 flex items-center justify-center z-10" 
                role="status" 
                aria-live="polite"
              >
                <div className="flex flex-col items-center">
                  <Loader2 size={32} className="animate-spin text-teal-500 mb-2" aria-hidden="true" />
                  <p className="text-teal-400 text-sm">Generating UMAP...</p>
                </div>
              </div>
            )}

            {points.length === 0 && !isLoading ? (
              <div 
                className="h-full flex flex-col items-center justify-center text-slate-500" 
                role="status" 
                aria-live="polite"
              >
                <Search size={32} className="mb-2" aria-hidden="true" />
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
                className="w-full h-full cursor-crosshair"
                onClick={handleCanvasClick}
                onMouseDown={handleCanvasMouseDown}
                onMouseMove={handleCanvasMouseMove}
                onMouseUp={handleCanvasMouseUp}
                onMouseLeave={handleCanvasMouseLeave}
                onWheel={handleWheel}
                aria-label="UMAP visualization canvas"
                role="img"
                tabIndex={0}
              />
            )}
            
            <div 
              className="absolute bottom-4 left-4 bg-slate-800/80 p-2 rounded text-xs text-white" 
              aria-live="polite"
            >
              <div>Tool: {currentTool === 'select' ? 'Selection' : 'Pan'}</div>
              <div>Zoom: {Math.round(scale * 100)}%</div>
              <div>Selected: {selectedPoints.length} points</div>
            </div>
          </div>
        </section>
        
        {isFullscreen && (
          <button
            onClick={toggleFullscreen}
            className="absolute top-4 right-4 p-2 bg-slate-800 rounded-full text-slate-300 hover:bg-slate-700"
            title="Exit Fullscreen"
            aria-label="Exit Fullscreen"
          >
            <X size={20} />
          </button>
        )}
      </div>
  );
}; 