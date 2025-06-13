import React, { createContext, useContext, useState, ReactNode, useEffect } from 'react';
import { VideoData, EventData } from '../types';

// EventData already defined in types.ts

// Define video context type
export interface VideoContextType {
  videos: VideoData[];
  events: EventData[];
  currentVideo: VideoData | null;
  currentTime: number;
  isPlaying: boolean;
  searchQuery: string;
  isLoading: boolean;
  error: string | null;
  filters: {
    category: string[];
    confidence: number;
    timeRange: [number, number];
  };
  // thêm filter cho events
  activeSearchFilters: {
    enableTextKeyword: boolean;
    textKeyword: string;
    minTextConfidence: number;
    enableObjectKeyword: boolean;
    objectKeyword: string;
    minObjectConfidence: number;
    enableClipSimilarity: boolean;
    minSimilarity: number;
  };
  setCurrentVideo: (video: VideoData | null) => void;
  setCurrentTime: (time: number) => void;
  setIsPlaying: (isPlaying: boolean) => void;
  setSearchQuery: (query: string) => void;
  updateFilters: (key: string, value: unknown) => void;
  updateActiveSearchFilters: (filters: {
    enableTextKeyword: boolean;
    textKeyword: string;
    minTextConfidence: number;
    enableObjectKeyword: boolean;
    objectKeyword: string;
    minObjectConfidence: number;
    enableClipSimilarity: boolean;
    minSimilarity: number;
  }) => void;
  addVideo: (video: VideoData) => void;
  searchByText: (query: string) => Promise<void>;
  searchByImage: (imageUrl: string) => Promise<void>;
  uploadVideo: (file: File) => Promise<void>;
  loadVideoEvents: (videoId: string) => Promise<void>;
  setEvents: (events: EventData[]) => void;
}

const VideoContext = createContext<VideoContextType | undefined>(undefined);

// Backend API URL
const API_BASE_URL = 'http://localhost:5000/api';

export const VideoProvider = ({ children }: { children: ReactNode }) => {
  const [videos, setVideos] = useState<VideoData[]>([]);
  const [events, setEvents] = useState<EventData[]>([]);
  const [currentVideo, setCurrentVideo] = useState<VideoData | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState({
    category: [],
    confidence: 0.1,
    timeRange: [0, 100] as [number, number],
  });
  
  // Thêm state activeSearchFilters
  const [activeSearchFilters, setActiveSearchFilters] = useState({
    enableTextKeyword: false,
    textKeyword: '',
    minTextConfidence: 0.8,
    enableObjectKeyword: false,
    objectKeyword: '',
    minObjectConfidence: 0.8,
    enableClipSimilarity: false,
    minSimilarity: 0.8,
  });

  // Fetch videos when component mounts
  useEffect(() => {
    fetchVideos();
  }, []);

  // Fetch all videos from backend
  const fetchVideos = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/videos`);
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      const data = await response.json();
      setVideos(data);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch videos';
      setError(errorMessage);
      console.error('Error fetching videos:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Load events for a specific video
  const loadVideoEvents = async (videoId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/video/${videoId}/events`);
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      const data = await response.json();
      setEvents(data);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch video events';
      setError(errorMessage);
      console.error('Error fetching video events:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Search events by text
  const searchByText = async (query: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          search_type: 'text',
          query,
          min_confidence: filters.confidence,
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      
      const data = await response.json();
      setEvents(data.events || []);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to search';
      setError(errorMessage);
      console.error('Error searching by text:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Search events by image
  const searchByImage = async (imageUrl: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          search_type: 'image',
          image_url: imageUrl,
          min_confidence: filters.confidence,
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      
      const data = await response.json();
      setEvents(data.events || []);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to search by image';
      setError(errorMessage);
      console.error('Error searching by image:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Upload video
  const uploadVideo = async (file: File) => {
    setIsLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('video', file);
      
      const response = await fetch(`${API_BASE_URL}/upload-video`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.status === 'success' && data.video) {
        setVideos(prev => [...prev, data.video]);
      }
      
      return data;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to upload video';
      setError(errorMessage);
      console.error('Error uploading video:', error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const updateFilters = (key: string, value: unknown) => {
    setFilters((prev) => ({
      ...prev,
      [key]: value,
    }));
  };
  //update filter cho event
  const updateActiveSearchFilters = (newFilters: {
    enableTextKeyword: boolean;
    textKeyword: string;
    minTextConfidence: number;
    enableObjectKeyword: boolean;
    objectKeyword: string;
    minObjectConfidence: number;
    enableClipSimilarity: boolean;
    minSimilarity: number;
  }) => {
    setActiveSearchFilters(newFilters);
  };
  
  const addVideo = (video: VideoData) => {
    setVideos((prev) => [...prev, video]);
  };

  return (
    <VideoContext.Provider
      value={{
        videos,
        events,
        currentVideo,
        currentTime,
        isPlaying,
        searchQuery,
        isLoading,
        error,
        filters,
        activeSearchFilters,
        setCurrentVideo,
        setCurrentTime,
        setIsPlaying,
        setSearchQuery,
        updateFilters,
        updateActiveSearchFilters,
        addVideo,
        searchByText,
        searchByImage,
        uploadVideo,
        loadVideoEvents,
        setEvents,
      }}
    >
      {children}
    </VideoContext.Provider>
  );
};

export const useVideo = () => {
  const context = useContext(VideoContext);
  if (context === undefined) {
    throw new Error('useVideo must be used within a VideoProvider');
  }
  return context;
};