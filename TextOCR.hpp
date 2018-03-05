//
//  TextOCR.hpp
//  MonsterScan
//
//  Created by n01192 on 9/15/16.
//  Copyright © 2016 n01192. All rights reserved.
//

#ifndef TextOCR_hpp
#define TextOCR_hpp

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/text.hpp>
#include <opencv2/imgproc.hpp>


class TextOCR{
    
    
public:
	TextOCR():m_regions(1){}
    void detect(cv::Mat& colrImage, cv::Mat& grayImage);
    std::vector<std::string> recognize(cv::Mat& grayImage);
    
    
    
private:
    
    
    std::vector<cv::Mat> m_channels;
    std::vector<std::vector<cv::text::ERStat>> m_regions;
    std::vector<std::vector<cv::Vec2i>> m_region_groups;
    std::vector<cv::Rect> m_group_boxes;
    std::vector<cv::Ptr<cv::text::ERFilter>> m_filter1;
    std::vector<cv::Ptr<cv::text::ERFilter>> m_filter2;
    
};

#endif /* TextOCR_hpp */
