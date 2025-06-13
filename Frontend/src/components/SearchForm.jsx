import React, { useState } from 'react';
import { Box, Button, FormControl, FormControlLabel, Checkbox, InputLabel, MenuItem, Select, Slider, TextField, Typography, Stack, Paper, Radio, RadioGroup, FormLabel } from '@mui/material';

const SearchForm = ({ onSearch, isLoading }) => {
  const [searchType, setSearchType] = useState('text');
  const [query, setQuery] = useState('');
  const [imageUrl, setImageUrl] = useState('');
  const [adaptiveThreshold, setAdaptiveThreshold] = useState(0.5);
  const [searchMethod, setSearchMethod] = useState('text_adaptive');
  const [topK, setTopK] = useState(10);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSearch({
      search_type: searchType,
      query,
      image_url: imageUrl,
      adaptive_threshold: adaptiveThreshold,
      search_method: searchMethod,
      top_k: topK
    });
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImageUrl(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
      <Box component="form" onSubmit={handleSubmit} noValidate>
        <Stack spacing={3}>
          <Typography variant="h6">Search Options</Typography>
          
          <FormControl fullWidth>
            <InputLabel>Search Type</InputLabel>
            <Select
              value={searchType}
              onChange={(e) => setSearchType(e.target.value)}
              label="Search Type"
            >
              <MenuItem value="text">Text Search</MenuItem>
              <MenuItem value="image">Image Search</MenuItem>
            </Select>
          </FormControl>

          {searchType === 'text' && (
            <>
              <TextField
                fullWidth
                label="Search Query"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter your search query"
                variant="outlined"
              />
              
              <FormControl component="fieldset">
                <FormLabel component="legend">Search Method</FormLabel>
                <RadioGroup
                  value={searchMethod}
                  onChange={(e) => setSearchMethod(e.target.value)}
                >
                  <FormControlLabel value="text_clip" control={<Radio />} label="Văn bản thuần (CLIP Similarity)" />
                  <FormControlLabel value="text_adaptive" control={<Radio />} label="Văn bản + Adaptive threshold" />
                  <FormControlLabel value="keyword_only" control={<Radio />} label="Chỉ keyword" />
                  <FormControlLabel value="text_keyword" control={<Radio />} label="Văn bản + keyword" />
                  <FormControlLabel value="object_only" control={<Radio />} label="Chỉ object" />
                  <FormControlLabel value="text_object" control={<Radio />} label="Văn bản + object" />
                  <FormControlLabel value="text_object_keyword" control={<Radio />} label="Văn bản + object + keyword" />
                </RadioGroup>
              </FormControl>
            </>
          )}

          {searchType === 'image' && (
            <Box>
              <Button
                variant="contained"
                component="label"
                sx={{ mb: 2 }}
              >
                Upload Image
                <input
                  type="file"
                  hidden
                  accept="image/*"
                  onChange={handleImageUpload}
                />
              </Button>
              
              {imageUrl && (
                <Box sx={{ mt: 2, mb: 2 }}>
                  <img 
                    src={imageUrl} 
                    alt="Upload Preview" 
                    style={{ maxWidth: '100%', maxHeight: '200px' }} 
                  />
                </Box>
              )}
              
              <TextField
                fullWidth
                label="Image URL (optional)"
                value={imageUrl}
                onChange={(e) => setImageUrl(e.target.value)}
                placeholder="Or enter image URL"
                variant="outlined"
                sx={{ mt: 2 }}
              />
            </Box>
          )}

          <Box>
            <Typography gutterBottom>
              Adaptive Threshold: {adaptiveThreshold}
            </Typography>
            <Slider
              value={adaptiveThreshold}
              onChange={(e, newValue) => setAdaptiveThreshold(newValue)}
              min={0}
              max={1}
              step={0.05}
              marks
              valueLabelDisplay="auto"
            />
          </Box>

          <FormControl fullWidth>
            <Typography gutterBottom>
              Maximum Results: {topK}
            </Typography>
            <Slider
              value={topK}
              onChange={(e, newValue) => setTopK(newValue)}
              min={5}
              max={50}
              step={5}
              marks
              valueLabelDisplay="auto"
            />
          </FormControl>

          <Button
            type="submit"
            variant="contained"
            color="primary"
            disabled={isLoading || (searchType === 'text' && !query) || (searchType === 'image' && !imageUrl)}
            fullWidth
          >
            {isLoading ? 'Searching...' : 'Search'}
          </Button>
        </Stack>
      </Box>
    </Paper>
  );
};

export default SearchForm; 