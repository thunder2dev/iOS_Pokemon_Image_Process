//
//  Descriptor.cpp
//  MonsterScan
//
//  Created by n01192 on 9/14/16.
//  Copyright © 2016 n01192. All rights reserved.
//

#include "Descriptor.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <opencv2/objdetect.hpp>
#include "Utils.hpp"
#include "HOG.hpp"



Descriptor::Descriptor(cv::Mat& image, const uint32_t label)
:m_label(label),
m_feaExtractor(cv::xfeatures2d::SIFT::create()){
    makeDesc(image);
    
}

Descriptor::Descriptor(NSString* filepath, uint32_t label)
:m_label(label){
    load(filepath);
}

Descriptor::~Descriptor(){
    m_hog.release();
    m_blobDesc.release();
    m_surfDesc.release();
}


uint32_t Descriptor::getLabel() const{
    return m_label;
}

cv::Mat Descriptor::getSurfFeature() const{
    return m_surfDesc;
}


cv::Mat Descriptor::getBlobFeature() const{
    return m_blobDesc;
}

cv::Mat Descriptor::getHogFeature() const{
    return m_hog;
}

void Descriptor::makeDesc(cv::Mat& image){
    
    /*if(!makeBlobDesc(image, m_blobDesc)){
        std::cout <<"blob extract faile"<<std::endl;
        return;
    }*/
    
    cv::GaussianBlur(image, image, cv::Size(3, 3), 0);
    
    cv::Mat gray;
    
    switch(image.channels()){
        case 4:
            cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
            break;
        case 3:
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            break;
    }

    /*
    if(!makeSurfDesc(gray, m_surfDesc)){
        std::cout <<"surf extract failed"<<std::endl;
        return;
    }*/

    
    
    if(!makeHogDesc(image, m_hog)){
        std::cout <<"surf extract failed"<<std::endl;
        return;
    }
    
    gray.release();
    
}


bool Descriptor::makeBlobDesc(cv::Mat& image, cv::Mat& desc){
    
    //return true;
    
    cv::Mat histogram;
    
	int histChannelNum = 8;

    int channels[] = {0, 1, 2};
    int numBins[] = { histChannelNum, histChannelNum, histChannelNum };
    
    float range[] = {0, 256};
    
    const float *ranges[] = {range, range, range};
    
    cv::calcHist(&image, 1, channels, cv::Mat(), histogram, 3, numBins, ranges);

    //histogram *= (1.0f/image.rows * image.cols);
    
	cv::Mat line = cv::Mat::zeros(cv::Size(histChannelNum*histChannelNum*histChannelNum, 1), CV_32FC1);

	for (int ii = 0; ii < histChannelNum; ii++) {
		for (int jj = 0; jj < histChannelNum; jj++) {
			for (int hh = 0; hh < histChannelNum; hh++) {
				line.at<float>(ii * histChannelNum*histChannelNum + jj * histChannelNum + hh) = histogram.at<float>(ii, jj, hh);
			}
		}
	}

    line.at<float>(0, 0) = 0;
    
	cv::normalize(line, desc, 1.0, 0, cv::NORM_L1);
    

	cv::Scalar sumline = cv::sum(desc);

    return true;
}

bool Descriptor::makeHogDesc(cv::Mat& image, cv::Mat& desc){
    
    //cv::Mat hogImage;
    
    //cv::resize(image, hogImage, cv::Size(64, 64));
    
    Utils::saveDebugImage(image, "hog image");
    
    //cv::HOGDescriptor hog(cv::Size(128, 64), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8),9);
    
    //int hsize = hog.getDescriptorSize();
    
    //std::vector<float> ders;
    
    //hog.compute(image,ders, cv::Size(16, 16));
    //hsize = hog.getDescriptorSize();
    //hog.save("/Users/n01192/Library/Developer/CoreSimulator/Devices/B384AFDB-FB5D-44C7-89AB-E31B35146B04/data/Containers/Data/Application/F11DDB57-CE0E-49A0-8CB1-1C44D279BD0A/Documents/aaaa.yml");
    
    /*
    
    m_hog = cv::Mat::zeros(ders.size(),1,CV_32FC1);
    
    for(int i=0;i<ders.size();i++)
    {
        m_hog.at<float>(i,0)=ders.at(i);
        
    }*/
    
    m_hog = extractHOG(image);
    
    return true;
}



bool Descriptor::makeSurfDesc(cv::Mat& gray, cv::Mat& desc){
    
    //return true;
    
    std::vector<cv::KeyPoint> keypoints;
    
    //cv::Mat blur;
    
    //cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);
    
    m_feaExtractor->detect(gray, keypoints);
    
    m_feaExtractor->compute(gray, keypoints, desc);
    
    
    return true;
}



void Descriptor::save(NSString* dir, int index){
    
    std::string dirC = [dir UTF8String];
    
    std::string path = dirC + "/" + std::to_string(m_label) + "-" + std::to_string(index) + ".yml";
    
    cv::FileStorage file(path, cv::FileStorage::WRITE);
    //file << "surf" << m_surfDesc;
    //file << "blob" << m_blobDesc;
    file << "hh" << m_hog;
    
}

void Descriptor::load(NSString* filepath){
    
    std::string pathC = [filepath UTF8String];
    
    cv::FileStorage file(pathC, cv::FileStorage::READ);
    //file["surf"] >> m_surfDesc;
    //file["blob"] >> m_blobDesc;
    file["hh"] >> m_hog;
    
}




















