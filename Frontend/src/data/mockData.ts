import { VideoData, EventData } from '../types';

export const mockVideos: VideoData[] = [
  {
    id: 'video-1',
    title: 'Traffic Intersection Morning',
    thumbnail: 'https://images.pexels.com/photos/1427541/pexels-photo-1427541.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    duration: 186,
    uploadDate: '2025-04-12',
    size: '256 MB',
    resolution: '1920x1080'
  },
  {
    id: 'video-2',
    title: 'City Surveillance Camera #23',
    thumbnail: 'https://images.pexels.com/photos/240225/pexels-photo-240225.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    duration: 294,
    uploadDate: '2025-04-10',
    size: '342 MB',
    resolution: '1920x1080'
  },
  {
    id: 'video-3',
    title: 'Retail Store Main Entrance',
    thumbnail: 'https://images.pexels.com/photos/264636/pexels-photo-264636.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    duration: 128,
    uploadDate: '2025-04-08',
    size: '184 MB',
    resolution: '1920x1080'
  }
];

export const mockEvents: EventData[] = [
  {
    id: 'event-1',
    videoId: 'video-1',
    title: 'Person Crossing',
    description: 'Individual crossing intersection at marked crosswalk',
    timestamp: 34,
    duration: 8,
    category: 'Pedestrian',
    confidence: 0.92,
    thumbnailUrl: 'https://images.pexels.com/photos/1486222/pexels-photo-1486222.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2'
  },
  {
    id: 'event-2',
    videoId: 'video-1',
    title: 'Vehicle Stopping',
    description: 'Red vehicle coming to a complete stop at intersection',
    timestamp: 62,
    duration: 5,
    category: 'Vehicle',
    confidence: 0.88,
    thumbnailUrl: 'https://images.pexels.com/photos/1209774/pexels-photo-1209774.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2'
  },
  {
    id: 'event-3',
    videoId: 'video-1',
    title: 'Bicycle Passing',
    description: 'Cyclist passing through intersection in bicycle lane',
    timestamp: 95,
    duration: 6,
    category: 'Bicycle',
    confidence: 0.79,
    thumbnailUrl: 'https://images.pexels.com/photos/853168/pexels-photo-853168.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2'
  },
  {
    id: 'event-4',
    videoId: 'video-2',
    title: 'Group Gathering',
    description: 'Group of individuals gathering near storefront',
    timestamp: 42,
    duration: 15,
    category: 'Group',
    confidence: 0.85,
    thumbnailUrl: 'https://images.pexels.com/photos/2774546/pexels-photo-2774546.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2'
  },
  {
    id: 'event-5',
    videoId: 'video-2',
    title: 'Delivery Vehicle',
    description: 'Delivery truck stopping in no-parking zone',
    timestamp: 128,
    duration: 12,
    category: 'Vehicle',
    confidence: 0.91,
    thumbnailUrl: 'https://images.pexels.com/photos/4391470/pexels-photo-4391470.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2'
  },
  {
    id: 'event-6',
    videoId: 'video-3',
    title: 'Customer Entry',
    description: 'Multiple customers entering retail location',
    timestamp: 28,
    duration: 9,
    category: 'Customer',
    confidence: 0.82,
    thumbnailUrl: 'https://images.pexels.com/photos/1102225/pexels-photo-1102225.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2'
  }
];

export const searchFilters = [
  {
    id: 'category',
    name: 'Event Category',
    type: 'select',
    options: ['Pedestrian', 'Vehicle', 'Bicycle', 'Group', 'Customer', 'All']
  },
  {
    id: 'confidence',
    name: 'Confidence Level',
    type: 'range',
    min: 0,
    max: 1
  },
  {
    id: 'timeRange',
    name: 'Time Range',
    type: 'range',
    min: 0,
    max: 100
  }
];