//
//  Classifier.hpp
//  MonsterScan
//
//  Created by n01192 on 9/15/16.
//  Copyright © 2016 n01192. All rights reserved.
//

#ifndef Classifier_hpp
#define Classifier_hpp

#include "Descriptor.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/ml.hpp>

class Classifier{
public:
    Classifier();
    
    void addDescriptor(Descriptor& desc);
    
    int nnPredict(const Descriptor& src);
    
    void load(std::string savePath);
    
    void svmTrain(cv::Mat& trainData, cv::Mat rsp, int nKind, std::string savePath);
    
    int svmPredict(cv::Mat& fea);
    
    int useless(cv::Mat& fea);

	std::vector<Descriptor> m_trainingDescs;

private:
    float calcDistance(const Descriptor &src, const Descriptor& dst);
    
    cv::Ptr<cv::DescriptorMatcher> m_feaMatcher;
    
    
    
    int m_nKind;
    
    cv::Ptr<cv::ml::SVM> m_svm;
    
    cv::ml::ParamGrid c_grid;
    cv::ml::ParamGrid gamma_grid;
    cv::ml::ParamGrid p_grid;
    cv::ml::ParamGrid nu_grid;
    cv::ml::ParamGrid coef_grid;
    cv::ml::ParamGrid degree_grid;
    
    void initSVMParams(const cv::Mat& rsp);
    void initGridParams();
    
    
    
    
};







#endif /* Classifier_hpp */
