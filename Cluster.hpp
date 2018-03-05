//
//  Cluster.hpp
//  MonsterS
//
//  Created by n01192 on 9/16/16.
//  Copyright © 2016 n01192. All rights reserved.
//

#ifndef Cluster_hpp
#define Cluster_hpp

#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgcodecs/ios.h>

class Cluster{
    
public:
    Cluster();
    
    void addDesc(cv::Mat& desc);
    void cluster();
    void save(NSString* dir);
    
private:
    cv::BOWKMeansTrainer m_bowTrainer;
    cv::Mat m_codebook;
    
};




#endif /* Cluster_hpp */
