import React, { useState } from 'react';
import { Search, Filter, X, Clock, BarChart, ChevronRight } from 'lucide-react';
import { useVideo } from '../context/VideoContext';
import { formatDuration } from '../utils/formatters';

export const EventList = () => {
  const { currentVideo, events, setCurrentTime } = useVideo();
  const [searchQuery, setSearchQuery] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState({
    category: '',
    confidence: 0.5,
    timeRange: [0, 100] as [number, number],
  });

  if (!currentVideo) {
    return (
      <div className="h-full flex items-center justify-center p-6">
        <p className="text-slate-500 text-center">
          Select a video to see detected events
        </p>
      </div>
    );
  }

  // Filter events for current video
  let videoEvents = events.filter((event) => event.videoId === currentVideo.id);

  // Apply search filter
  if (searchQuery) {
    videoEvents = videoEvents.filter(
      (event) =>
        event.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        event.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
        event.category.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }

  // Apply category filter
  if (filters.category) {
    videoEvents = videoEvents.filter((event) => event.category === filters.category);
  }

  // Apply confidence filter
  videoEvents = videoEvents.filter((event) => event.confidence >= filters.confidence);

  // Apply time range filter
  const videoDuration = currentVideo.duration;
  const startTime = (filters.timeRange[0] / 100) * videoDuration;
  const endTime = (filters.timeRange[1] / 100) * videoDuration;
  videoEvents = videoEvents.filter(
    (event) => event.timestamp >= startTime && event.timestamp <= endTime
  );

  const handleEventClick = (timestamp: number) => {
    setCurrentTime(timestamp);
  };

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value);
  };

  const handleClearSearch = () => {
    setSearchQuery('');
  };

  const updateFilters = (key: string, value: any) => {
    setFilters((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  return (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b border-slate-700">
        <h2 className="text-lg font-medium mb-3">Detected Events</h2>
        
        {/* Search Bar */}
        <div className="relative mb-3">
          <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
            <Search size={16} className="text-slate-400" />
          </div>
          <input
            type="text"
            value={searchQuery}
            onChange={handleSearchChange}
            placeholder="Search events..."
            className="w-full py-2 pl-10 pr-10 bg-slate-700 border border-slate-600 rounded-lg text-sm focus:ring-teal-500 focus:border-teal-500"
          />
          {searchQuery && (
            <button
              onClick={handleClearSearch}
              className="absolute inset-y-0 right-0 flex items-center pr-3 text-slate-400 hover:text-slate-200"
            >
              <X size={16} />
            </button>
          )}
        </div>

        {/* Filter Button */}
        <div className="flex items-center justify-between">
          <p className="text-sm text-slate-400">
            {videoEvents.length} events found
          </p>
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-sm ${
              showFilters
                ? 'bg-teal-600 border-teal-500 text-white'
                : 'border-slate-600 text-slate-400 hover:text-white hover:border-slate-500'
            }`}
          >
            <Filter size={14} />
            Filters
          </button>
        </div>

        {/* Filters Panel */}
        {showFilters && (
          <div className="mt-3 p-3 bg-slate-700 rounded-lg space-y-3">
            <div>
              <label className="block text-sm font-medium mb-1">
                Event Category
              </label>
              <select
                value={filters.category}
                onChange={(e) => updateFilters('category', e.target.value)}
                className="w-full p-2 bg-slate-600 border border-slate-500 rounded text-sm"
              >
                <option value="">All Categories</option>
                <option value="Pedestrian">Pedestrian</option>
                <option value="Vehicle">Vehicle</option>
                <option value="Bicycle">Bicycle</option>
                <option value="Group">Group</option>
                <option value="Customer">Customer</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">
                Minimum Confidence: {Math.round(filters.confidence * 100)}%
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={filters.confidence}
                onChange={(e) =>
                  updateFilters('confidence', parseFloat(e.target.value))
                }
                className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-teal-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">
                Time Range: {filters.timeRange[0]}% - {filters.timeRange[1]}%
              </label>
              <div className="flex items-center gap-2">
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={filters.timeRange[0]}
                  onChange={(e) =>
                    updateFilters('timeRange', [
                      parseInt(e.target.value),
                      filters.timeRange[1],
                    ])
                  }
                  className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-teal-500"
                />
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={filters.timeRange[1]}
                  onChange={(e) =>
                    updateFilters('timeRange', [
                      filters.timeRange[0],
                      parseInt(e.target.value),
                    ])
                  }
                  className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-teal-500"
                />
              </div>
            </div>

            <div className="flex justify-end gap-2">
              <button
                onClick={() => {
                  updateFilters('category', '');
                  updateFilters('confidence', 0.5);
                  updateFilters('timeRange', [0, 100]);
                }}
                className="px-3 py-1.5 border border-slate-500 rounded text-sm hover:bg-slate-600"
              >
                Reset
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {videoEvents.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center p-6">
            <div className="w-12 h-12 rounded-full bg-slate-700 flex items-center justify-center mb-4">
              <BarChart size={24} className="text-slate-500" />
            </div>
            <h3 className="text-lg font-medium mb-2">No Events Found</h3>
            <p className="text-slate-400">
              {searchQuery || filters.category || filters.confidence > 0.5
                ? 'Try adjusting your search or filters.'
                : 'This video doesn\'t have any detected events yet.'}
            </p>
          </div>
        ) : (
          videoEvents.map((event) => (
            <div
              key={event.id}
              className="bg-slate-800 rounded-lg overflow-hidden cursor-pointer hover:bg-slate-750 transition-colors duration-200"
              onClick={() => handleEventClick(event.timestamp)}
            >
              <div className="relative aspect-video">
                <img
                  src={event.thumbnailUrl}
                  alt={event.title}
                  className="w-full h-full object-cover"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-transparent">
                  <div className="absolute bottom-2 left-2 flex items-center text-white text-xs bg-black/50 px-2 py-1 rounded">
                    <Clock size={12} className="mr-1" />
                    {formatDuration(event.timestamp)}
                  </div>
                  <div
                    className="absolute top-2 right-2 text-xs px-2 py-1 rounded"
                    style={{
                      backgroundColor: getCategoryColor(event.category),
                    }}
                  >
                    {event.category}
                  </div>
                </div>
              </div>

              <div className="p-3">
                <div className="flex justify-between items-start mb-1">
                  <h3 className="font-medium">{event.title}</h3>
                  <div className="flex items-center text-xs text-teal-400">
                    <span>View</span>
                    <ChevronRight size={14} />
                  </div>
                </div>
                <p className="text-sm text-slate-400 mb-2">{event.description}</p>
                <div className="flex items-center gap-3">
                  <div className="flex-1">
                    <div className="text-xs text-slate-400 mb-1">Confidence</div>
                    <div className="w-full h-1.5 bg-slate-700 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${event.confidence * 100}%`,
                          backgroundColor: getConfidenceColor(event.confidence),
                        }}
                      ></div>
                    </div>
                  </div>
                  <div className="text-sm font-medium">
                    {Math.round(event.confidence * 100)}%
                  </div>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
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
  };
  return colors[category] || '#9CA3AF';
};

const getConfidenceColor = (confidence: number) => {
  if (confidence >= 0.9) return '#10B981'; // Green
  if (confidence >= 0.7) return '#F59E0B'; // Amber
  return '#EF4444'; // Red
};