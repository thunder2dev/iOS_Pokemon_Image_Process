//
//  ImagePreProcessor.cpp
//  MonsterScan
//
//  Created by n01192 on 9/15/16.
//  Copyright © 2016 n01192. All rights reserved.
//

#include "ImagePreProcessor.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include "Const.h"
#include "Utils.hpp"
#include <stack>


ImagePreProcessor::ImagePreProcessor(cv::Mat& image)
:m_image(image),
m_hasWhiteBorder(false){
    
    resize();
/*
    if(m_image_ori.channels() == 3){
        
        cv::cvtColor(m_image_ori, m_image, cv::COLOR_RGB2BGR);
    }else{
        cv::cvtColor(m_image_ori, m_image, cv::COLOR_RGBA2BGR);
    }
  */
    cv::cvtColor(m_image, m_gray, cv::COLOR_BGR2GRAY);
    
    if(getWhiteBordRect()){
        
        m_hasWhiteBorder = true;
        
    }
    
}

ImagePreProcessor::~ImagePreProcessor(){
    m_image.release();
    m_gray.release();
    m_green.release();
    m_white.release();
    m_bottomWhite.release();
    
    m_normImage.release();
    m_normGray.release();
    m_pockerRoi.release();
    
    
}


bool ImagePreProcessor::getContourMat(cv::Mat& image, cv::Mat& contourMat){
    
    cv::Mat contours;
    
    Canny(image, contours, 125, 350);
    
    cv::Mat contourInv;
    
    cv::threshold(contours, contourInv, 128, 255, cv::THRESH_BINARY_INV);
    
    contourMat = contourInv;
    
    return true;
    
}



void seedFill(const cv::Mat& _binImg, cv::Mat& _lableImg, std::vector<int>& labels, std::vector<int>& areas)
{
    // connected component analysis (4-component)
    // use seed filling algorithm
    // 1. begin with a foreground pixel and push its foreground neighbors into a stack;
    // 2. pop the top pixel on the stack and label it with the same label until the stack is empty
    //
    // foreground pixel: _binImg(x,y) = 1
    // background pixel: _binImg(x,y) = 0
    
    
    if (_binImg.empty() ||
        _binImg.type() != CV_8UC1)
    {
        return ;
    }
    
    _lableImg.release() ;
    _binImg.convertTo(_lableImg, CV_32SC1) ;
    
    int label = 1 ;  // start by 2
    
    int rows = _binImg.rows - 1 ;
    int cols = _binImg.cols - 1 ;
    
    cv::Rect rect(0, 0, cols - 1, rows - 1);
    
    
    for (int i = 1; i < rows-1; i++)
    {
        int* data= _lableImg.ptr<int>(i) ;
        for (int j = 1; j < cols-1; j++)
        {
            if (data[j] == 1)
            {
                std::stack<std::pair<int,int>> neighborPixels ;
                neighborPixels.push(std::pair<int,int>(i,j)) ;     // pixel position: <i,j>
                ++label ;  // begin with a new label
                int area = 0;
                
                while (!neighborPixels.empty())
                {
                    // get the top pixel on the stack and label it with the same label
                    std::pair<int,int> curPixel = neighborPixels.top() ;
                    int curX = curPixel.first ;
                    int curY = curPixel.second ;
                    _lableImg.at<int>(curX, curY) = label ;
                    area += 1;
                    
                    // pop the top pixel
                    neighborPixels.pop() ;
                    
                    // push the 4-neighbors (foreground pixels)
                    if (rect.contains(cv::Point(curY - 1, curX)) && _lableImg.at<int>(curX, curY-1) == 1)
                    {// left pixel
                        neighborPixels.push(std::pair<int,int>(curX, curY-1)) ;
                    }
                    if (rect.contains(cv::Point(curY+1,curX)) && _lableImg.at<int>(curX, curY+1) == 1)
                    {// right pixel
                        neighborPixels.push(std::pair<int,int>(curX, curY+1)) ;
                    }
                    if (rect.contains(cv::Point(curY, curX-1)) && _lableImg.at<int>(curX-1, curY) == 1)
                    {// up pixel
                        neighborPixels.push(std::pair<int,int>(curX-1, curY)) ;
                    }
                    if (rect.contains(cv::Point(curY, curX+1)) && _lableImg.at<int>(curX+1, curY) == 1)
                    {// down pixel
                        neighborPixels.push(std::pair<int,int>(curX+1, curY)) ;
                    }
                }
                labels.push_back(label);
                areas.push_back(area);
            }  
        }  
    }  
}


static void floodFillPostprocess( cv::Mat& img, const cv::Scalar& colorDiff=cv::Scalar::all(1) )
{
    CV_Assert( !img.empty() );
    cv::RNG rng = cv::theRNG();
    cv::Mat mask( img.rows+2, img.cols+2, CV_8UC1, cv::Scalar::all(0) );
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            if( mask.at<uchar>(y+1, x+1) == 0 )
            {
                cv::Scalar newVal( rng(256), rng(256), rng(256) );
                floodFill( img, mask, cv::Point(x,y), newVal, 0, colorDiff, colorDiff );
            }
        }
    }
}

void cutBackground(cv::Mat& image, cv::Mat& mask, cv::Mat& outMask){
    int channels_b[] = {0};
    int channels_g[] = {1};
    int channels_r[] = {2};
    int histSize[] = {256};
    float range[] = {0, 256};
    const float *ranges[] = {range};
    cv::Mat hist;
    int backR, backG, backB;
    
    cv::calcHist(&image, 1, channels_b, mask, hist, 1, histSize, ranges);
    
    cv::Point maxIdx, minIdx;
    double minVal, maxVal;
    
    cv::minMaxLoc(hist, &minVal, &maxVal, &minIdx, &maxIdx);
    
    backB = static_cast<int>(maxIdx.y);
    
    cv::calcHist(&image, 1, channels_g, mask, hist, 1, histSize, ranges);
    
    cv::minMaxLoc(hist, &minVal, &maxVal, &minIdx, &maxIdx);
    
    backG = static_cast<int>(maxIdx.y);
    
    cv::calcHist(&image, 1, channels_r, mask, hist, 1, histSize, ranges);
    
    cv::minMaxLoc(hist, &minVal, &maxVal, &minIdx, &maxIdx);
    
    backR = static_cast<int>(maxIdx.y);
    
    cv::Mat backRange;
    cv::inRange(image, cv::Scalar(backB-20, backG-20, backR-20), cv::Scalar(backB+20, backG+20, backR+20), backRange);
    
    int backChannels = backRange.channels();
    
    outMask = backRange;
    
    Utils::saveDebugImage(outMask, "back range");
}


bool ImagePreProcessor::getPockerRoi(cv::Mat& roi){
    
    cv::Rect pockerRoi;
    
    cf::getPockerRoi(pockerRoi);
    
    roi = m_image(pockerRoi);//.copyTo(roi);
    
    Utils::saveDebugImage(roi, "roi");
    return true;
    
    
    cv::Mat blurImage;
    
    cv::GaussianBlur(m_image, blurImage, cv::Size(5, 5), 0);
    
    cv::Mat topHalf;
    
    topHalf = blurImage(cv::Rect(0, 0, blurImage.cols, m_whiteBorder.tl().y - 5));//.copyTo(topHalf);
    
    cv::Mat miniHalf;
    
    cv::resize(topHalf, miniHalf, cv::Size(static_cast<int>(topHalf.cols*0.3), static_cast<int>(topHalf.rows*0.3)));
    
    
    Utils::saveDebugImage(miniHalf, "mini top half");
    
    cv::Mat meanShift;
    
    cv::pyrMeanShiftFiltering(miniHalf, meanShift, 30, 30, 1);
    
    Utils::saveDebugImage(meanShift, "mean shift");
    
    floodFillPostprocess( meanShift, cv::Scalar::all(10) );
    
    Utils::saveDebugImage(meanShift, "mean shift fill");
    
    cv::Mat meanMask = cv::Mat::zeros(meanShift.rows, meanShift.cols, CV_8UC1);
    
    meanMask.colRange(0, 20) = 255;
    
    Utils::saveDebugImage(meanMask, "mean mask");
    
    cv::Mat backRange, backLeft, backRight;
    
    ::cutBackground(meanShift, meanMask, backLeft);
    //backLeft.colRange(backLeft.cols/2, backLeft.cols) = 0;
    
    meanMask.colRange(0, 20) = 0;
    meanMask.colRange(meanMask.cols - 20, meanMask.cols) = 255;
    
    ::cutBackground(meanShift, meanMask, backRight);
    //backRight.colRange(0, backRight.cols/2) = 0;
    
    cv::bitwise_or(backLeft, backRight, backRange);
    
    Utils::saveDebugImage(backRange, "back Range or");
    
    backRange = 255 - backRange;
    
    Utils::saveDebugImage(backRange, "back Range not");
    
    cv::Mat backClose;
    
    cv::morphologyEx(backRange, backClose, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10, 10)));
    
    Utils::saveDebugImage(backClose, "back close");
    
    cv::Mat backDilate;
    
    cv::dilate(backClose, backDilate, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(30, 30)));
    
    Utils::saveDebugImage(backDilate, "back dilate");
    
    cv::Mat backDst;
    
    cv::bitwise_and(backDilate, backRange, backDst);
    
    cv::Mat bigMask;
    
    cv::resize(backDst, bigMask, cv::Size(topHalf.cols, topHalf.rows));
    
    bigMask = bigMask > 0;
    
    Utils::saveDebugImage(bigMask, "big mask");
    
    cv::Mat bigCloseTinyMask;
    
    cv::morphologyEx(bigMask, bigCloseTinyMask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
    
    Utils::saveDebugImage(bigCloseTinyMask, "big close tiny mask");
    
    cv::Mat fullMask;
    
    cv::copyMakeBorder(bigCloseTinyMask, fullMask, 0, blurImage.rows - bigMask.rows, 0, 0, cv::BORDER_CONSTANT, 0);
    
    fullMask.rowRange(bigCloseTinyMask.rows + 8, fullMask.rows) = 255;
    
    Utils::saveDebugImage(fullMask, "full mask");
    
    cv::Mat fullDilateMask;
    
    cv::morphologyEx(fullMask, fullDilateMask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(12, 12)));
    
    Utils::saveDebugImage(fullDilateMask, "full dilate mask");
    
    cv::Mat fullDialteOpenMask;
    
    cv::morphologyEx(fullDilateMask, fullDialteOpenMask,cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15)));
    
    Utils::saveDebugImage(fullDialteOpenMask, "full dilate open mask");
    
    /*
    cv::Mat contour;
    
    cv::Canny(blurImage, contour, 20, 50);
    
    Utils::saveDebugImage(blurImage, "blur image");
    */
     
    //cv::Mat openWhite = m_bottomWhite;
    
    //cv::Mat notWhite;
    
    //cv::bitwise_and(m_image, 255-openWhite, notWhite);
    
    cv::Mat image;
    
    cv::copyMakeBorder(m_image, image, 0, 0, 0, 0, cv::BORDER_DEFAULT);
    
    //image = m_image;//.copyTo(image);
    
    image.setTo(cv::Scalar(0, 0, 0), m_bottomWhite);
    
    Utils::saveDebugImage(image, "not white");
    
    //cv::bitwise_and(notWhite, fullMask, roi);
    
    image.setTo(cv::Scalar(0, 0, 0), 255-fullDialteOpenMask);
    
    Utils::saveDebugImage(image, "roi image");
    
    roi = image(pockerRoi);//.copyTo(roi);
    
    Utils::saveDebugImage(roi, "roi");
    
    /*
    cv::Mat centerContourImage;
    cv::Mat centerImage;
    
    contour(pockerRoi).copyTo(centerContourImage);
    contour(pockerRoi).copyTo(centerImage);

    Utils::saveDebugImage(centerContourImage, "center contour");*/
    
    
    

    

	return true;
}



/*
bool ImagePreProcessor::getPockerRoiAndNum(cv::Mat& roi, float& num){
    
    cv::Mat bigRoi;
    m_image(cv::Rect(0, 0, m_image.cols , (int)(m_image.rows * POKER_VERT_PER))).copyTo(bigRoi);

    cv::Mat noBgRoi;
    
    cutBackground(bigRoi, noBgRoi);
    
    Utils::saveDebugImage(noBgRoi, "big roi grab cut");
    
    cv::Mat contourImage;

    cv::Canny(bigRoi, contourImage, 20, 50);

    Utils::saveDebugImage(contourImage, "big roi contour");
    
    cv::Mat mo = cv::Mat::zeros(5, 5, CV_8UC1);
    
    mo.row(2) = 255;
    
    cv::morphologyEx(contourImage, contourImage, cv::MORPH_OPEN, mo);
    
    mo = cv::Mat::zeros(5, 5, CV_8UC1);
    
    mo.col(2) = 255;
    
    cv::morphologyEx(contourImage, contourImage, cv::MORPH_OPEN, mo);
    
    cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10, 10));
    
    cv::morphologyEx(contourImage, contourImage, cv::MORPH_CLOSE, mo);
    
    cv::Rect maxRect;
    
    if(!getMaxContourRect(contourImage, maxRect)){
        
        NSLog(@"get pocker roi failed");
        
    }
    return true;
}*/


bool ImagePreProcessor::resize(){
    
    if(m_image.cols < MAX_WIDTH && m_image.rows < MAX_HEIGHT){
        return false;
    }
    
    
    float ratio = 1.0 * m_image.cols / m_image.rows;
    float ratio_ref = 1.0 * MAX_WIDTH/MAX_HEIGHT;
    
    if(ratio > ratio_ref){
        
        m_width = MAX_WIDTH;
        m_height = MAX_WIDTH / ratio;
		
        
    }else{
        
        m_height = MAX_HEIGHT;
        m_width = m_height * ratio;
        
    }
    
    cv::resize(m_image, m_image, cv::Size(m_width, m_height), 0, 0, cv::INTER_CUBIC);
    


    return true;
}



bool ImagePreProcessor::getTinyCircle(cv::Mat& grayImage, cv::Point grayTl, cv::Point bigCenter, float bigR, float tinyR, cv::Point& center, int& area){
    
    /*
    std::vector<cv::Vec3f> circles;
    
    cv::HoughCircles(grayImage, circles, cv::HOUGH_GRADIENT, 2, 1, 40, 80, 0, 0);
    
    cv::Mat grayContour;
    cv::Canny(grayImage, grayContour, 100, 100);
    Utils::saveDebugImage(grayContour, "gray contour");
    
    int bigROffset = 10;
    int tinyROffset = 3;
    
    for(auto iter = circles.begin(); iter != circles.end(); ++iter){
        
        cv::Point ori = cv::Point((*iter)[0], (*iter)[1]);
        int radius = (*iter)[2];
        
        auto bigOff = ori + grayTl - bigCenter;
        
        float bigOffNorm = cv::norm(bigOff);
        
        if(fabs(bigOffNorm - bigR) > bigROffset){
            continue;
        }
        
        if(fabs(radius - tinyR) > tinyROffset){
            continue;
        }
        
        center = ori + grayTl;
        
        return true;
        

    }*/
    
    cv::Mat erodeDotImage;
    
    cv::morphologyEx(grayImage, erodeDotImage, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(8, 8)));
  
    Utils::saveDebugImage(erodeDotImage, "erode dot");
    
    //auto sum = cv::sum(erodeDotImage);
    //float currS = 1.0*sum[0]/255.0;
    float currS = cv::countNonZero(erodeDotImage);
    float minS = CV_PI * tinyR * tinyR *0.75;
    
    if(currS < minS){
        return false;
    }
    
    cv::Rect dotRect;
    
    getMaxContourRect(erodeDotImage, dotRect);
    
    center = dotRect.tl() + cv::Point(dotRect.width/2, dotRect.height/2);
    
    center += grayTl;
    
    area = currS;
    
    erodeDotImage.release();
    
    
    return true;
}



bool ImagePreProcessor::getWhiteBordRect(){
    
    cv::Mat whiteImage;
    
    cv::inRange(m_image, cv::Scalar(240, 240, 240), cv::Scalar(255, 255, 255), whiteImage);
    
    cv::copyMakeBorder(whiteImage, m_white, 0, 0, 0, 0, cv::BORDER_DEFAULT);
    
    //m_white = whiteImage;//.copyTo(m_white);
    
    int kernelWidth = 30;
    
    cv::Size kernelSize(kernelWidth, kernelWidth);
    
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, kernelSize);
    
    cv::morphologyEx(whiteImage, whiteImage, cv::MORPH_OPEN, kernel);

    if(!getMaxContourRect(whiteImage, m_whiteBorder)){
        return false;
    }
    
    //std::vector<std::vector<cv::Point>> contours;
    //std::vector<cv::Vec4i> hierarchy;
    
    //cv::findContours(whiteImage, contours, hierarchy, CV_RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    Utils::saveDebugImage(m_white, "white");
    
    cv::Mat labelImage;
    std::vector<int> labels;
    std::vector<int> areas;
    
    m_bottomWhite = whiteImage/255;
    
    seedFill(m_bottomWhite, labelImage, labels, areas);
    
    int max = INT_MIN, maxIdx = 1;
    
    for(int ii = 0; ii < areas.size(); ii++){
        if(areas[ii] > max){
            max = areas[ii];
            maxIdx = ii;
        }
    }
    
    m_bottomWhite = labelImage == labels[maxIdx];
    
    Utils::saveDebugImage(m_bottomWhite, "bottom white");
    
    
    /*
    for(int ii = 0; ii < contours.size(); ii++){
        
        cv::Rect rect = cv::boundingRect(contours[ii]);
        
        if(rect.area() < m_whiteBorder.area()*0.7){
        
            auto contour = contours[ii];
            auto hier = hierarchy[ii];
            //if(hier[3] < 0){
                cv::drawContours(m_white, contours, ii, cv::Scalar(0, 0, 0), CV_FILLED, 8, hierarchy, 2);
                Utils::saveDebugImage(m_white, "white delete little above");
            //}
        }
    }
    
    Utils::saveDebugImage(m_white, "white delete little above");
    */
    
    return true;
    
}


bool ImagePreProcessor::getGreenLine(cv::Point& tll, int& widthh, bool isGreen) {
	
	cv::Mat greenRoi;
    greenRoi = m_image(m_whiteBorder);//.copyTo(greenRoi);

    Utils::saveDebugImage(greenRoi, "green roi");
    
	cv::Mat green;
	cv::Scalar greenColor(187, 237, 109);
    if(!isGreen){
        greenColor = cv::Scalar(204, 204, 204);
    }
    
	cv::Scalar greenOffset(40, 40, 40);

	cv::inRange(greenRoi, greenColor - greenOffset, greenColor + greenOffset, green);

    Utils::saveDebugImage(green, "green");
    
	cv::Mat grayOpen;
	cv::morphologyEx(green, grayOpen, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    /*
    cv::Mat grayClose;
    cv::morphologyEx(grayOpen, grayClose, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(6, 6)));*/

	cv::Mat grayLines;

	cv::Canny(grayOpen, grayLines, 50, 200);
    
    Utils::saveDebugImage(grayLines, "gray lines");

	std::vector<cv::Vec4i> lines;
	cv::HoughLinesP(grayLines, lines, 2, CV_PI / 90, 150, 50, 2);

    if(lines.size()< 2){
        return false;
    }
    
    //cv::Mat draw;
    //greenRoi.copyTo(draw);
    
    cv::Mat mask = cv::Mat::zeros(grayOpen.rows, grayOpen.cols, CV_8UC1);
    
    int maxWidth = 0;
    cv::Point maxTl(10000, 10000);
    
	// draw lines
	for (size_t i = 0; i < lines.size(); i++)
	{
        cv::Point left(lines[i][0], lines[i][1]);
        cv::Point right(lines[i][2], lines[i][3]);
        
        cv::Point lineOff = right.y > left.y ? right - left: left - right;
        
        float lineAng = atan2(lineOff.y, lineOff.x) * 180/CV_PI;
        
        if(lineAng> 30 && lineAng < 150){
            continue;
        }
        
        cv::Point tl = left + m_whiteBorder.tl();
        float width = right.x - left.x;
        
        if(i != 0 && width < 0.8 * maxWidth){
            continue;
        }
        
        //cv::line(draw, left, right, cv::Scalar(255, 0, 0));
        
        cv::line(mask, left, right, cv::Scalar(255, 0, 0), 20);
        
        cv::Mat greenOr;
        
        cv::bitwise_and(mask, green, greenOr);
        
        cv::Mat greenOrHist;
        
        cv::reduce(greenOr, greenOrHist, 0, CV_REDUCE_MAX);
        
        greenOrHist = greenOrHist/255;
        
        width = cv::sum(greenOrHist)[0];
        
        if(i == 0){
            maxWidth = width;
            maxTl = tl;
        }else if(width > 0.8 * maxWidth && tl.y < maxTl.y){
            maxTl = tl;
            if(width > maxWidth){
                maxWidth = width;
            }
        }
        
        tll = maxTl;
        widthh = maxWidth;
        
        //Utils::saveDebugImage(draw, "draw");
        Utils::saveDebugImage(greenOr, "draw green or");
        
        greenOr.release();
        greenOrHist.release();
                
	}

    greenRoi.release();
    green.release();
    grayOpen.release();
    grayLines.release();
    mask.release();
	//cv::imshow("detected lines", grayLines);

	return true;
}


std::vector<cv::Rect> ImagePreProcessor::getContourRects(cv::Mat& wbImage){
    
    std::vector<cv::Rect> rects;
    
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    
    cv::findContours(wbImage, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    
    for(auto contour: contours){
        cv::Rect rect = cv::boundingRect(contour);
        rects.push_back(rect);
    }
    return rects;
    
}


bool ImagePreProcessor::getMaxContourRect(cv::Mat& wbImage, cv::Rect& maxRect){
    
    cv::Mat tmpImage;// = wbImage;//.copyTo(tmpImage);
    
    cv::copyMakeBorder(wbImage, tmpImage, 0, 0, 0, 0, cv::BORDER_DEFAULT);
    
    std::vector<cv::Rect> rects = getContourRects(tmpImage);
    
    bool found = false;
    float max = std::numeric_limits<float>::lowest();
    
    for(auto rect: rects){
        
        float area = rect.width * rect.height;
        
        if(area > max){
            max = area;
            maxRect = rect;
            found = true;
        }
    }
    
    tmpImage.release();
    
    return found;
    
}


bool ImagePreProcessor::getWatchNum(float& num, std::vector<float> lvAngs, cv::Point& numCenter, float& numR, cv::Point& watchCenter){
    
    cv::Mat circleGrayImage;
	cv::Mat circleWhiteImage;
    
    circleGrayImage= m_gray(cv::Rect(0, 0, m_image.cols, m_whiteBorder.tl().y));//.copyTo(circleGrayImage);
    circleWhiteImage = m_white(cv::Rect(0, 0, m_image.cols, m_whiteBorder.tl().y));//.copyTo(circleWhiteImage);
    
    
    //cv::Mat draw = cv::Mat::zeros(circleGrayImage.rows, circleGrayImage.cols, CV_8UC1);

	float watchR;

	cf::getWatchCenter(watchCenter, watchR);

	//cv::ellipse(draw, watchCenter, cv::Size(watchR, watchR), 0, 180, 360, cv::Scalar(255), 30);

    //Utils::saveDebugImage(draw, "draw ellipse");
    
    //cv::Mat tinyDotImage;
    
    //cv::bitwise_and(circleWhiteImage, draw, tinyDotImage);
    
    //Utils::saveDebugImage(tinyDotImage, "draw bit and");
    
    
    /*
    cv::Mat debugDraw;
    circleGrayImage.copyTo(debugDraw);
    cv::ellipse(debugDraw, watchCenter, cv::Size(watchR, watchR), 0, 180, 360, cv::Scalar(100), 1);
     */
    
    /*
    float dotCos = (dotPos.x - center.x)/ R;
    
    float dotAngPi = std::acos(dotCos);
    
    float dotAng = dotAngPi/3.14159265 * 180;
    
    if(dotAng < 0){
        dotAng = 90 - dotAng;
    }
    
    num = dotAng/180;
    
	numCenter = center;
	numR = R;
    */

    int totSucc = 0;
    int maxArea = 0;
    float maxAng = 0;
    float maxR = 0;
    cv::Point maxCenter;
    
    
    int sliceR = static_cast<int>(1.0f * watchR * CV_PI * (5.0/180.0) /2.0);
    
    std::vector<float> allAngs;
    //lvAngs = std::vector<float>();
    
    if(lvAngs.size() <= 0){
        for(int ii =360 ; ii >= 180; ii-=3){
            allAngs.push_back(ii);
        }
    }else{
        for(int ii = 0; ii < lvAngs.size(); ii++){
            allAngs.push_back(360 - lvAngs[ii]);
        }
    }

    
    
    
    for(int jj = 0; jj < allAngs.size(); jj++){
        
        float ii = allAngs[jj];
        
        cv::Mat mask = cv::Mat::zeros(circleWhiteImage.rows, circleWhiteImage.cols, CV_8UC1);
        
        cv::ellipse(mask, watchCenter, cv::Size(watchR, watchR), 0, ii-2.5, ii+2.5, cv::Scalar(255), 30);
        
        float raid = 1.0 * (360.0 -ii)/180.0 * CV_PI;
        int sliceRelX = static_cast<int>(cos(raid) * watchR);
        int sliceRelY = static_cast<int>(-fabs(sin(raid) * watchR));
        
        cv::Point sliceCenter = watchCenter + cv::Point(sliceRelX, sliceRelY);
        cv::Point sliceTl = sliceCenter + cv::Point(-sliceR, -sliceR);
        
        cv::Mat sliceImage;
        
        cv::Rect sliceRoi(sliceTl.x, sliceTl.y, sliceR*2, sliceR*2);
        
        sliceImage = circleWhiteImage(sliceRoi);//.copyTo(sliceImage);
        
        Utils::saveDebugImage(sliceImage, "slice" + std::to_string(ii));

        //cv::Mat sliceSum;
        
        auto sliceSum = cv::sum(sliceImage);
        
        
        if(sliceSum[0]> 1){
            cv::Point tinyCenter;
            int area;
            if(!getTinyCircle(sliceImage, sliceTl,watchCenter, watchR, cf::getTinyR(), tinyCenter, area)){
                if(totSucc >= 1){
                    num = maxAng;
                    numCenter = maxCenter;
                    numR = maxR;
                    
                    //cv::line(debugDraw, watchCenter, numCenter, cv::Scalar(100));
                    //Utils::saveDebugImage(debugDraw, "draw debug watch");
                    mask.release();
                    sliceImage.release();
                    circleGrayImage.release();
                    circleWhiteImage.release();
                    return true;
                }
            }else{
                cv::Point tinyOff(tinyCenter.x - watchCenter.x, watchCenter.y - tinyCenter.y);
                
                if(totSucc == 0 || (totSucc == 1 && area > maxArea)){
                    
                    float ang = atan2(tinyOff.y, tinyOff.x) * 180/CV_PI;
                    
                    if(ang < 0){
                        if(ang < -90){
                            ang = 180;
                        }else{
                            ang = 0;
                        }
                    }
                    
                    
                    maxArea = area;
                    maxAng = ang;
                    maxCenter = tinyCenter;
                    maxR = watchR;
                    totSucc += 1;
                    
                }
                
                if(totSucc == 2 || (totSucc == 1 && ii == 180) || (jj == allAngs.size() - 1)){
                    num = maxAng;
                    numCenter = maxCenter;
                    numR = maxR;
                    
                    //cv::line(debugDraw, watchCenter, numCenter, cv::Scalar(100));
                    //Utils::saveDebugImage(debugDraw, "draw debug watch");
                    mask.release();
                    sliceImage.release();
                    circleGrayImage.release();
                    circleWhiteImage.release();
                    return true;
                }
                
            }
        }else if(totSucc >= 1){
            num = maxAng;
            numCenter = maxCenter;
            numR = maxR;
            
            //cv::line(debugDraw, watchCenter, numCenter, cv::Scalar(100));
            //Utils::saveDebugImage(debugDraw, "draw debug watch");
            mask.release();
            sliceImage.release();
            circleGrayImage.release();
            circleWhiteImage.release();
            return true;
        }
        
        mask.release();
        sliceImage.release();
    }
    
    
    circleGrayImage.release();
    circleWhiteImage.release();
    
    return false;
}



bool ImagePreProcessor::cutBackground(cv::Mat& image, cv::Mat& res){
    
    cv::Rect rect(20, 40, image.cols -20, image.rows - 20);
    
    cv::Mat bgModel, fgModel;
    
    cv::Mat ress = cv::Mat::ones(image.rows, image.cols, CV_8UC1);
    
    cv::grabCut(image, ress, rect, bgModel, fgModel, 1, cv::GC_INIT_WITH_RECT);
    
    res = ress;
    
    return true;
    
}


























