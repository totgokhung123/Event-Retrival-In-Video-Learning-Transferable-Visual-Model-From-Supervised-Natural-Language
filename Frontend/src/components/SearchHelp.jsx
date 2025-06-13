import React, { useState } from 'react';
import { 
  Paper, 
  Typography, 
  Accordion, 
  AccordionSummary, 
  AccordionDetails,
  Box
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

const SearchHelp = () => {
  const [expanded, setExpanded] = useState(false);

  const handleChange = (panel) => (event, isExpanded) => {
    setExpanded(isExpanded ? panel : false);
  };

  const searchMethods = [
    {
      id: 'text_clip',
      name: 'Văn bản thuần (CLIP Similarity)',
      description: 'Tìm kiếm bằng cách so sánh ngữ nghĩa giữa văn bản và hình ảnh sử dụng mô hình CLIP. Tìm các khung hình có nội dung tương tự với văn bản tìm kiếm, không áp dụng ngưỡng lọc.'
    },
    {
      id: 'text_adaptive',
      name: 'Văn bản + Adaptive threshold',
      description: 'Tương tự như văn bản thuần, nhưng chỉ trả về kết quả có điểm tương đồng cao hơn ngưỡng adaptive threshold. Giúp lọc bỏ kết quả không liên quan.'
    },
    {
      id: 'keyword_only',
      name: 'Chỉ keyword',
      description: 'Tìm kiếm chính xác các khung hình có chứa từ khóa đã nhập trong văn bản được phát hiện (OCR). Hữu ích khi cần tìm một từ hoặc cụm từ cụ thể.'
    },
    {
      id: 'text_keyword',
      name: 'Văn bản + keyword',
      description: 'Kết hợp tìm kiếm ngữ nghĩa và tìm kiếm từ khóa. Tăng điểm cho khung hình xuất hiện trong cả hai phương pháp. Cân bằng giữa độ chính xác và ngữ cảnh.'
    },
    {
      id: 'object_only',
      name: 'Chỉ object',
      description: 'Tìm kiếm các khung hình có chứa đối tượng được đề cập trong văn bản tìm kiếm. Hữu ích khi cần tìm một đối tượng cụ thể như "xe", "người", v.v.'
    },
    {
      id: 'text_object',
      name: 'Văn bản + object',
      description: 'Kết hợp tìm kiếm ngữ nghĩa và tìm kiếm đối tượng. Tăng điểm cho khung hình chứa cả ngữ cảnh và đối tượng cần tìm.'
    },
    {
      id: 'text_object_keyword',
      name: 'Văn bản + object + keyword',
      description: 'Kết hợp cả ba phương pháp, đạt hiệu quả cao nhất khi cần tìm kiếm phức tạp. Tăng điểm cho khung hình xuất hiện trong nhiều phương pháp. Phù hợp cho truy vấn phức tạp.'
    }
  ];

  return (
    <Paper elevation={2} sx={{ p: 2, mb: 3 }}>
      <Typography variant="h6" gutterBottom>
        Hướng dẫn phương pháp tìm kiếm
      </Typography>
      
      <Box sx={{ mt: 2 }}>
        {searchMethods.map((method) => (
          <Accordion 
            key={method.id}
            expanded={expanded === method.id}
            onChange={handleChange(method.id)}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography fontWeight={expanded === method.id ? 'bold' : 'normal'}>
                {method.name}
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body2" color="text.secondary">
                {method.description}
              </Typography>
            </AccordionDetails>
          </Accordion>
        ))}
      </Box>
      
      <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
        Adaptive Threshold: Điều chỉnh ngưỡng độ tin cậy cho kết quả. Giá trị cao hơn sẽ giảm số lượng kết quả nhưng tăng độ chính xác.
      </Typography>
    </Paper>
  );
};

export default SearchHelp; 