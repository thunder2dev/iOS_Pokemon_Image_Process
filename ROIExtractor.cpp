//
//  ROIExtractor.cpp
//  MonsterScan
//
//  Created by n01192 on 9/14/16.
//  Copyright © 2016 n01192. All rights reserved.
//

#include <iostream>
#include "ROIExtractor.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xphoto.hpp>
#include "ImagePreProcessor.hpp"


bool ROIExtractor::simpleExtract(cv::Mat &image, cv::Mat& roi){
    
    cv::Mat mask;
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    
    
    
    if(!getBiMask(gray, mask)){
        std::cout <<" get bi failed"<<std::endl;
        return false;
    }
    
    cv::Mat edges;
    cv::Canny(mask, edges, 190, 255);
    
    
    cv::Mat edgesClose;
    cv::morphologyEx(edges, edgesClose, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10, 10)));
    
    
    cv::Rect maxRect;
    
    if(!ImagePreProcessor::getMaxContourRect(edgesClose, maxRect)){
        return false;
    }
    
    roi = image(maxRect);//.copyTo(roi);
    
    return true;
}


bool ROIExtractor::getBiMask(const cv::Mat& image, cv::Mat& mask){
    
    cv::Mat meanColor;
    cv::Mat stdColor;
    
    cv::meanStdDev(image, meanColor, stdColor);
    
    cv::threshold(image, mask, 200, 255, cv::THRESH_BINARY_INV);
        
    int kernelWidth = 20;
    
    cv::Size kernelSize(kernelWidth, kernelWidth);
    
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, kernelSize);
    
    cv::erode(mask, mask, kernel, cv::Point(-1, -1), 1);
    
    return true;
}

