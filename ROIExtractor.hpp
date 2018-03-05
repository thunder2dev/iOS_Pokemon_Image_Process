//
//  ROIExtractor.hpp
//  MonsterScan
//
//  Created by n01192 on 9/14/16.
//  Copyright © 2016 n01192. All rights reserved.
//

#ifndef ROIExtractor_hpp
#define ROIExtractor_hpp

#include <opencv2/core/core.hpp>

class ROIExtractor{
public:
    
    static bool simpleExtract(cv::Mat &image, cv::Mat& roi);
    
    
private:
    
    static bool getBiMask(const cv::Mat& image, cv::Mat& mask);
    
};


#endif /* ROIExtractor_hpp */
