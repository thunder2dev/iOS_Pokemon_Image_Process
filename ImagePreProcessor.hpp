//
//  ImagePreProcessor.hpp
//  MonsterScan
//
//  Created by n01192 on 9/15/16.
//  Copyright © 2016 n01192. All rights reserved.
//

#ifndef ImagePreProcessor_hpp
#define ImagePreProcessor_hpp

#include <opencv2/core/core.hpp>

class ImagePreProcessor{
    
    
public:
    ImagePreProcessor(cv::Mat& image);
    
    bool getPockerRoi(cv::Mat& roi);
    
    bool getWatchNum(float& num, std::vector<float> lvArgs, cv::Point& numCenter, float& numR, cv::Point& watchCenter);

	bool getGreenLine(cv::Point& tl, int& width, bool green) ;
    
    static std::vector<cv::Rect> getContourRects(cv::Mat& wbImage);
    static bool getMaxContourRect(cv::Mat& wbImage, cv::Rect& maxRect);
    
    ~ImagePreProcessor();
    
	cv::Mat m_image;
	cv::Mat m_gray;
	cv::Mat m_green;
	cv::Mat m_white;
    cv::Mat m_bottomWhite;
private:
    
    
    bool resize();
    
    bool getContourMat(cv::Mat& image, cv::Mat& mat);
    
    bool getTinyCircle(cv::Mat& grayImage, cv::Point grayTl, cv::Point bigCenter, float bigR, float tinyR, cv::Point& center, int& area);
    

    bool cutBackground(cv::Mat& image, cv::Mat& res);
    
    
    bool getWhiteBordRect();
    
	const float POCKER_START_ANG = 45;
	const float POCKER_END_ANG = 135;
    
    const float ROI_WIDTH = 700;
    
    const float CENTER_PERMIT_OFFSET = 20;
    
    const float POKER_VERT_PER = 0.4f;
    const float MAX_WIDTH = 1000;
    const float MAX_HEIGHT = 2000;
    
        
    cv::Mat m_normImage;
    cv::Mat m_normGray;
    cv::Mat m_pockerRoi;
    
    
    bool m_hasWhiteBorder;

    cv::Rect m_whiteBorder;
    
    float m_width;
    float m_height;
    
    
};



#endif /* ImagePreProcessor_hpp */