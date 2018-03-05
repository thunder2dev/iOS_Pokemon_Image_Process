//
//  Descriptor.hpp
//  MonsterScan
//
//  Created by n01192 on 9/14/16.
//  Copyright © 2016 n01192. All rights reserved.
//

#ifndef Descriptor_hpp
#define Descriptor_hpp

#import <opencv2/imgcodecs/ios.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>

class Descriptor{

public:
    Descriptor(NSString* filepath, uint32_t label);
    Descriptor(cv::Mat& image, const uint32_t label);
    ~Descriptor();
    
    void makeDesc(cv::Mat& image);
    
    cv::Mat getSurfFeature() const;
    cv::Mat getBlobFeature() const;
    cv::Mat getHogFeature() const;
    
    void save(NSString* dir, int index);
    void load(NSString* filepath);
    
    uint32_t getLabel() const;
    
    
private:
    
    uint32_t m_label;
    
    cv::Mat m_surfDesc;
    cv::Mat m_blobDesc;
    cv::Mat m_hog;
    
    
    bool makeSurfDesc(cv::Mat& image, cv::Mat& desc);
    bool makeBlobDesc(cv::Mat& image, cv::Mat& desc);
    bool makeHogDesc(cv::Mat& image, cv::Mat& desc);
    
    
    cv::Ptr<cv::Feature2D> m_feaExtractor;

    
};


#endif /* Descriptor_hpp */
