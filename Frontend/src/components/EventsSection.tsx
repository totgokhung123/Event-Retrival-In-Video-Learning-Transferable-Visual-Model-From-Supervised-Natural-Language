import React, { useState } from 'react';
import { Clock, BarChart, ChevronRight, Eye, Search, Zap, Filter, Tag } from 'lucide-react';
import { useVideo } from '../context/VideoContext';
import { formatDuration } from '../utils/formatters';

const API_BASE_URL = 'http://localhost:5000/api';

// Mở rộng EventData để có thêm các trường confidence
interface ExtendedEventData {
  id: string;
  videoId: string;
  title: string;
  description: string;
  timestamp: number;
  duration: number;
  category: string;
  confidence: number;
  thumbnailUrl: string;
  // Thêm các trường confidence đặc biệt
  text_confidence?: number;
  object_confidence?: number;
  clip_similarity?: number;
  detection_type?: string;
}

export const EventsSection = () => {
  const { currentVideo, events, setCurrentTime, isLoading, activeSearchFilters } = useVideo();

  // Local filter states for display control
  const [displayFilters, setDisplayFilters] = useState({
    showTextDetection: true,
    showObjectDetection: true,
    showClipSimilarity: true,
    showLabels: true, // Đổi tên từ enableObjectDetection thành showLabels
  });

  const handleEventClick = (timestamp: number) => {
    setCurrentTime(timestamp);
  };

  // Get confidence values from event data for different detection types
  const getEventConfidences = (event: ExtendedEventData) => {
    // Lấy trực tiếp các giá trị confidence từ dữ liệu JSON
    const mainConfidence = event.confidence || 0;
    const textConfidence = event.text_confidence || 0;
    const objectConfidence = event.object_confidence || 0;
    const clipConfidence = event.clip_similarity || 0;
    
    return {
      text: textConfidence,
      object: objectConfidence,
      clip: clipConfidence,
      main: mainConfidence
    };
  };

  // Determine which confidence values to display based on available data and search filters
  const getDisplayConfidences = (event: ExtendedEventData) => {
    const confidenceValues = getEventConfidences(event);
    const confidences = [];
    
    // Xác định phương thức tìm kiếm đang sử dụng
    const usingTextKeyword = activeSearchFilters.enableTextKeyword;
    const usingObjectKeyword = activeSearchFilters.enableObjectKeyword;
    const usingClipSimilarity = activeSearchFilters.enableClipSimilarity;
    
    // Text Detection - chỉ hiển thị khi đang sử dụng Text Keyword
    if (usingTextKeyword && displayFilters.showTextDetection) {
      confidences.push({
        value: confidenceValues.text,
        label: 'Text Detection',
        color: getConfidenceColor(confidenceValues.text),
        bgColor: 'from-teal-500 to-cyan-500'
      });
    }
    
    // Object Detection - chỉ hiển thị khi đang sử dụng Object Keyword
    if (usingObjectKeyword && displayFilters.showObjectDetection && displayFilters.showLabels) {
      confidences.push({
        value: confidenceValues.object,
        label: 'Object Detection',
        color: getConfidenceColor(confidenceValues.object),
        bgColor: 'from-purple-500 to-pink-500'
      });
    }
    
    // CLIP Similarity - hiển thị khi:
    // 1. Người dùng bật filter CLIP Similarity, HOẶC
    // 2. Khi detection_type là "clip", HOẶC
    // 3. Khi không sử dụng keyword hoặc object keyword nhưng có clip_similarity > 0
    const hasClipSimilarity = confidenceValues.clip > 0;
    const isClipDetection = event.detection_type === "clip";
    
    if ((usingClipSimilarity || isClipDetection || (hasClipSimilarity && !usingTextKeyword && !usingObjectKeyword)) 
        && displayFilters.showClipSimilarity) {
      confidences.push({
        value: confidenceValues.clip,
        label: 'CLIP Similarity',
        color: getConfidenceColor(confidenceValues.clip),
        bgColor: 'from-orange-500 to-red-500'
      });
    }
    
    // Nếu không có confidence nào được hiển thị, hiện confidence chính
    if (confidences.length === 0) {
      confidences.push({
        value: confidenceValues.main,
        label: 'Confidence',
        color: getConfidenceColor(confidenceValues.main),
        bgColor: 'from-slate-500 to-slate-600'
      });
    }
    
    return confidences;
  };
  
  const updateDisplayFilter = (key: string, value: boolean) => {
    setDisplayFilters((prev) => ({
      ...prev,
      [key]: value
    }));
  };
  
  // Tạo URL cho hình ảnh thumbnail
  const getThumbnailUrl = (thumbnailPath?: string) => {
    if (!thumbnailPath) return '';
    if (thumbnailPath.startsWith('http')) return thumbnailPath;
    
    const filename = thumbnailPath.split(/[/\\]/).pop();
    return `${API_BASE_URL}/frame/${filename}`;
  };

  // Custom toggle component
  const ConfidenceToggle = ({ 
    checked, 
    onChange, 
    label, 
    icon, 
    color = 'teal',
    disabled = false 
  }: {
    checked: boolean;
    onChange: (checked: boolean) => void;
    label: string;
    icon: React.ReactNode;
    color?: string;
    disabled?: boolean;
  }) => {
    const colorClasses = {
      teal: 'from-teal-500 to-cyan-500',
      purple: 'from-purple-500 to-pink-500',
      orange: 'from-orange-500 to-red-500',
      blue: 'from-blue-500 to-indigo-500'
    };

    return (
      <div className={`flex items-center gap-2 ${disabled ? 'opacity-50' : ''}`}>
        <button
          type="button"
          onClick={() => !disabled && onChange(!checked)}
          disabled={disabled}
          className={`relative w-10 h-5 rounded-full transition-all duration-300 ${
            checked && !disabled
              ? `bg-gradient-to-r ${colorClasses[color as keyof typeof colorClasses]} shadow-lg`
              : 'bg-slate-600'
          } ${!disabled ? 'cursor-pointer' : 'cursor-not-allowed'}`}
        >
          <div
            className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow-md transition-all duration-300 ${
              checked ? 'translate-x-5' : 'translate-x-0.5'
            }`}
          />
        </button>
        <div className="flex items-center gap-1.5">
          {icon}
          <span className={`text-sm font-medium ${disabled ? 'text-slate-500' : 'text-slate-300'}`}>
            {label}
          </span>
        </div>
      </div>
    );
  };

  // Check if any search filters are active
  const hasActiveSearchFilters = Object.values(activeSearchFilters).some(value => 
    typeof value === 'boolean' ? value : false
  );

  if (isLoading) {
    return (
      <div className="p-6 flex items-center justify-center h-64">
        <div className="flex flex-col items-center">
          <div className="w-12 h-12 border-4 border-slate-700 border-t-teal-500 rounded-full animate-spin mb-4"></div>
          <p className="text-slate-400">Đang tải dữ liệu...</p>
        </div>
      </div>
    );
  }

  if (!currentVideo && events.length === 0) {
    return (
      <div className="h-full flex items-center justify-center p-6">
        <p className="text-slate-500 text-center">
          Chọn một video hoặc tìm kiếm để xem các sự kiện phát hiện được
        </p>
      </div>
    );
  }

  return (
    <div className="p-6">
      {/* Header with Labels Master Toggle */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-xl font-medium mb-2">Detected Events Timeline</h2>
            <p className="text-sm text-slate-400">
              {events.length} events detected
              {hasActiveSearchFilters && (
                <span className="ml-2 text-teal-400">• Advanced filters applied</span>
              )}
            </p>
          </div>

          {/* Show Labels Master Toggle */}
          <div className="flex items-center gap-3 bg-slate-800/50 backdrop-blur-sm rounded-xl p-3 border border-slate-700/50">
            <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-blue-500/20 to-indigo-500/20 flex items-center justify-center">
              <Tag size={14} className="text-blue-400" />
            </div>
            <ConfidenceToggle
              checked={displayFilters.showLabels}
              onChange={(checked) => updateDisplayFilter('showLabels', checked)}
              label="Show Labels"
              icon={<></>}
              color="blue"
            />
          </div>
        </div>

        {/* Confidence Display Filters */}
        {hasActiveSearchFilters && (
          <div className="bg-slate-800/30 backdrop-blur-sm rounded-xl p-4 border border-slate-700/30 mb-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-slate-500/20 to-slate-600/20 flex items-center justify-center">
                <Filter size={14} className="text-slate-400" />
              </div>
              <h3 className="text-sm font-semibold text-slate-300">Confidence Display Filters</h3>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Text Detection Toggle - Luôn hiển thị */}
              <ConfidenceToggle
                checked={displayFilters.showTextDetection}
                onChange={(checked) => updateDisplayFilter('showTextDetection', checked)}
                label="Text Detection"
                icon={<Eye size={14} className="text-teal-400" />}
                color="teal"
                disabled={!activeSearchFilters.enableTextKeyword}
              />
              
              {/* Object Detection Toggle - Luôn hiển thị */}
              <ConfidenceToggle
                checked={displayFilters.showObjectDetection}
                onChange={(checked) => updateDisplayFilter('showObjectDetection', checked)}
                label="Object Detection"
                icon={<Search size={14} className="text-purple-400" />}
                color="purple"
                disabled={!activeSearchFilters.enableObjectKeyword || !displayFilters.showLabels}
              />
              
              {/* CLIP Similarity Toggle - Luôn hiển thị và có thể tương tác */}
              <ConfidenceToggle
                checked={displayFilters.showClipSimilarity}
                onChange={(checked) => updateDisplayFilter('showClipSimilarity', checked)}
                label="CLIP Similarity"
                icon={<Zap size={14} className="text-orange-400" />}
                color="orange"
              />
            </div>
            
            {/* Quick Actions */}
            <div className="flex justify-end gap-2 mt-4 pt-3 border-t border-slate-700/30">
              <button
                onClick={() => {
                  setDisplayFilters(prev => ({
                    ...prev,
                    showTextDetection: true,
                    showObjectDetection: true,
                    showClipSimilarity: true,
                  }));
                }}
                className="px-3 py-1.5 text-xs bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-all duration-200 text-slate-300"
              >
                Show All
              </button>
              <button
                onClick={() => {
                  setDisplayFilters(prev => ({
                    ...prev,
                    showTextDetection: false,
                    showObjectDetection: false,
                    showClipSimilarity: false,
                  }));
                }}
                className="px-3 py-1.5 text-xs bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-all duration-200 text-slate-300"
              >
                Hide All
              </button>
            </div>
          </div>
        )}
      </div>

      {events.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-64 text-center p-6">
          <div className="w-16 h-16 rounded-full bg-slate-700 flex items-center justify-center mb-4">
            <BarChart size={32} className="text-slate-500" />
          </div>
          <h3 className="text-xl font-medium mb-2">No Events Detected</h3>
          <p className="text-slate-400">
            Try adjusting your search criteria or select a different video.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {events.map((event) => {
            const confidences = getDisplayConfidences(event as ExtendedEventData);
            
            return (
              <div
                key={event.id}
                className="bg-slate-800 rounded-xl overflow-hidden cursor-pointer hover:bg-slate-750 transition-all duration-200 hover:scale-105 hover:shadow-xl"
                onClick={() => handleEventClick(event.timestamp)}
              >
                <div className="relative aspect-video">
                  <img
                    src={getThumbnailUrl(event.thumbnailUrl)}
                    alt={event.title}
                    className="w-full h-full object-cover"
                    crossOrigin="anonymous"
                    onError={(e) => {
                      // Fallback nếu ảnh không tải được
                      const target = e.target as HTMLImageElement;
                      target.src = 'https://via.placeholder.com/640x360/1e293b/334155?text=No+Image';
                    }}
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-transparent">
                    <div className="absolute bottom-3 left-3 flex items-center text-white text-sm bg-black/60 px-3 py-1.5 rounded-lg">
                      <Clock size={14} className="mr-2" />
                      {formatDuration(event.timestamp)}
                    </div>
                    {displayFilters.showLabels && (
                      <div
                        className="absolute top-3 right-3 text-sm px-3 py-1.5 rounded-lg font-medium"
                        style={{
                          backgroundColor: getCategoryColor(event.category),
                        }}
                      >
                        {event.category}
                      </div>
                    )}
                  </div>
                </div>

                <div className="p-4">
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="font-medium text-lg">{event.title}</h3>
                    <div className="flex items-center text-sm text-teal-400">
                      <span>View</span>
                      <ChevronRight size={16} className="ml-1" />
                    </div>
                  </div>
                  <p className="text-sm text-slate-400 mb-4 line-clamp-2">{event.description}</p>
                  
                  {/* Multiple Confidence Displays */}
                  <div className="space-y-3">
                    {confidences.map((confidence, index) => (
                      <div key={index} className="flex items-center gap-3">
                        <div className="flex-1">
                          <div className="flex items-center justify-between mb-1">
                            <div className="text-xs text-slate-400 font-medium">{confidence.label}</div>
                            <div className="text-sm font-medium">
                              {Math.round(confidence.value * 100)}%
                            </div>
                          </div>
                          <div className="w-full h-1.5 bg-slate-700 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full bg-gradient-to-r ${confidence.bgColor} transition-all duration-300`}
                              style={{
                                width: `${confidence.value * 100}%`,
                              }}
                            ></div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

const getCategoryColor = (category: string) => {
  const colors: Record<string, string> = {
    Pedestrian: '#3B82F6',
    Vehicle: '#F59E0B',
    Bicycle: '#10B981',
    Group: '#6366F1',
    Customer: '#EC4899',
    Unknown: '#9CA3AF',
    person: '#3B82F6',
    car: '#F59E0B',
    bicycle: '#10B981',
    tv: '#6366F1'
  };
  return colors[category] || '#9CA3AF';
};

const getConfidenceColor = (confidence: number) => {
  if (confidence >= 0.9) return '#10B981'; // Green
  if (confidence >= 0.7) return '#F59E0B'; // Amber
  return '#EF4444'; // Red
};