//
//  Cluster.cpp
//  MonsterS
//
//  Created by n01192 on 9/16/16.
//  Copyright © 2016 n01192. All rights reserved.
//

#include "Cluster.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>

using namespace std;
using namespace cv;


Cluster::Cluster()
:m_bowTrainer(BOWKMeansTrainer(200))
{
}


void Cluster::addDesc(cv::Mat &desc){  //desc hoz orientation
    
    m_bowTrainer.add(desc);
    
    
    
    
}


void Cluster::cluster(){
    
    m_codebook = m_bowTrainer.cluster();
    

    
    
}

void Cluster::save(NSString* dir){
    std::string dirC = [dir UTF8String];
    
    std::string path = dirC + "/codebook.yml";
    
    cv::FileStorage file(path, cv::FileStorage::WRITE);
    file << "codebook" << m_codebook;
}