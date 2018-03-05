//
//  TextOCR.cpp
//  MonsterScan
//
//  Created by n01192 on 9/15/16.
//  Copyright © 2016 n01192. All rights reserved.
//

#include "TextOCR.hpp"


using namespace std;
using namespace cv;
using namespace cv::text;

void TextOCR::detect(cv::Mat& colrImage, cv::Mat& grayImage){
    
	m_channels.push_back(grayImage);

    cv::Ptr<text::ERFilter> er_filter1 = createERFilterNM1(loadClassifierNM1("C:/Users/Administrator/Desktop/MonsterTrainer/MonsterTrainer/x64/Debug/trained_classifierNM1.xml"), 16, 0.00015f, 0.13f, 0.2f, true, 0.1f);
    
    cv::Ptr<text::ERFilter> er_filter2 = createERFilterNM2(loadClassifierNM2("C:/Users/Administrator/Desktop/MonsterTrainer/MonsterTrainer/x64/Debug/trained_classifierNM2.xml"), 0.5);

    for(int c = 0; c < (int)m_channels.size(); c++){
        er_filter1->run(m_channels[c], m_regions[c]);
        er_filter2->run(m_channels[c], m_regions[c]);
    }
    
    
    erGrouping(colrImage, m_channels, m_regions, m_region_groups, m_group_boxes, ERGROUPING_ORIENTATION_HORIZ);
    
    er_filter1.release();
    er_filter2.release();
    
    // m_channels.push_back(grayImage);
    // m_channels.push_back(255 - grayImage);
    
    
}


std::vector<std::string> TextOCR::recognize(cv::Mat& grayImage){
    
    
    vector<Mat> detections;
    
    for(int ii = 0; ii < (int)m_group_boxes.size(); ii++){
        Mat group_img = Mat::zeros(grayImage.rows+2, grayImage.cols+2, CV_8UC1);
        vector<Vec2i>& group = m_region_groups[ii];
        for(int jj = 0; jj < (int)group.size(); jj++){
            
            ERStat& er = m_regions[group[jj][0]][group[jj][1]];
            if(er.parent != NULL){
                
                int newMaskVal = 255;
                int flag = 4 + (newMaskVal<<8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
                
                cv::Mat& channel = m_channels[group[jj][0]];
                
                cv::floodFill(channel, group_img, Point(er.pixel%channel.cols, er.pixel/channel.cols), Scalar(255), 0, Scalar(er.level), Scalar(0), flag);
                
            }
            
        }
        
        group_img(m_group_boxes[ii]).copyTo(group_img);
        cv::copyMakeBorder(group_img, group_img, 15, 15, 15, 15, BORDER_CONSTANT, Scalar(0));
        detections.push_back(group_img);
        
    }
    
    
    vector<string> outputs((int)detections.size());
    vector<vector<Rect>> boxes((int)detections.size());
    vector<vector<string>> words((int)detections.size());
    vector<vector<float>> confidences((int)detections.size());
    
    Ptr<OCRTesseract> tesser = OCRTesseract::create();
    
    
    for(int i = 0; i < (int)detections.size(); i++){
        
        tesser->run(detections[i], outputs[i]);
        //(detections[i], outputs[i], &boxes[i], &words[i], &confidences, OCR_LEVEL_WORD);
        outputs[i].erase(remove(outputs[i].begin(), outputs[i].end(), '\n'), outputs[i].end());
        
        
        
        
    }
    
    return outputs;
    
    
    
    
}






























