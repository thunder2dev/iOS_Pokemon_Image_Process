//
//  Coder.hpp
//  MonsterS
//
//  Created by n01192 on 9/16/16.
//  Copyright © 2016 n01192. All rights reserved.
//

#ifndef Coder_hpp
#define Coder_hpp

#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>


class Coder{
    
public:
    
    Coder();
    
    void setCodeBook(cv::Mat& codebook);
    
    cv::Mat code(cv::Mat& input);
    
    cv::Mat llccode(cv::Mat &codebook, cv::Mat &input, cv::Mat IDX, int k);
    
    cv::Mat coeff2code(cv::Mat &coeff);
    
private:
    cv::Mat brutalFindKNN(cv::Mat &codebook, cv::Mat &input, int k);
    
    
    cv::Mat m_codebook;
    cv::flann::Index m_flannIdx;
    
};



#endif /* Coder_hpp */
