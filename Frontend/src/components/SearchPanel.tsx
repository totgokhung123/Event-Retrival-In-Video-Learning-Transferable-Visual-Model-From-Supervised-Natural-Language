import React, { useState } from 'react';
import { Search, Filter, X } from 'lucide-react';
import { useVideo } from '../context/VideoContext';
import { searchFilters } from '../data/mockData';

export const SearchPanel = () => {
  const { searchQuery, setSearchQuery, filters, updateFilters } = useVideo();
  const [showFilters, setShowFilters] = useState(false);

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value);
  };

  const toggleFilters = () => {
    setShowFilters(!showFilters);
  };

  const handleClearSearch = () => {
    setSearchQuery('');
  };

  return (
    <div className="relative">
      <div className="flex items-center">
        <div className="relative">
          <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
            <Search size={16} className="text-slate-400" />
          </div>
          <input
            type="text"
            value={searchQuery}
            onChange={handleSearchChange}
            placeholder="Search events..."
            className="py-2 pl-10 pr-10 bg-slate-700 border border-slate-600 rounded-lg text-sm focus:ring-teal-500 focus:border-teal-500 w-64"
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
        <button
          onClick={toggleFilters}
          className={`ml-2 p-2 rounded-lg border ${
            showFilters
              ? 'bg-teal-600 border-teal-500 text-white'
              : 'border-slate-600 text-slate-400 hover:text-white hover:border-slate-500'
          }`}
        >
          <Filter size={16} />
        </button>
      </div>

      {showFilters && (
        <div className="absolute right-0 mt-2 w-80 bg-slate-800 border border-slate-700 rounded-lg shadow-lg z-50 p-4">
          <div className="flex justify-between items-center mb-4">
            <h3 className="font-medium">Filters</h3>
            <button
              onClick={toggleFilters}
              className="text-slate-400 hover:text-white"
            >
              <X size={16} />
            </button>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">
                Event Category
              </label>
              <select
                value={filters.category[0] || ''}
                onChange={(e) => updateFilters('category', [e.target.value])}
                className="w-full p-2 bg-slate-700 border border-slate-600 rounded text-sm"
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
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-teal-500"
              />
              <div className="flex justify-between text-xs text-slate-400 mt-1">
                <span>0%</span>
                <span>50%</span>
                <span>100%</span>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">
                Time Range: {formatTimeRange(filters.timeRange)}
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
                  className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-teal-500"
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
                  className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-teal-500"
                />
              </div>
              <div className="flex justify-between text-xs text-slate-400 mt-1">
                <span>Start</span>
                <span>End</span>
              </div>
            </div>

            <div className="pt-2 flex justify-end gap-2">
              <button
                onClick={() => {
                  updateFilters('category', []);
                  updateFilters('confidence', 0.5);
                  updateFilters('timeRange', [0, 100]);
                }}
                className="px-3 py-1.5 border border-slate-600 rounded text-sm hover:bg-slate-700"
              >
                Reset
              </button>
              <button
                onClick={toggleFilters}
                className="px-3 py-1.5 bg-teal-600 rounded text-sm text-white hover:bg-teal-500"
              >
                Apply
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const formatTimeRange = (range: [number, number]) => {
  return `${range[0]}% - ${range[1]}%`;
};