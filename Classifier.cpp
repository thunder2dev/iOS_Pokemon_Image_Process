//
//  Classifier.cpp
//  MonsterScan
//
//  Created by n01192 on 9/15/16.
//  Copyright © 2016 n01192. All rights reserved.
//

#include "Classifier.hpp"
#include <opencv2/imgproc.hpp>


using namespace std;
using namespace cv;
using namespace cv::ml;

Classifier::Classifier():
m_feaMatcher(cv::DescriptorMatcher::create("FlannBased"))
{
    
}

void Classifier::addDescriptor(Descriptor& desc){
    
    m_trainingDescs.push_back(desc);
    
    
}


int Classifier::nnPredict(const Descriptor& src){
    
    float minDist = FLT_MAX;
    float minLabel = -1;
    
    
    for(const Descriptor& dst: m_trainingDescs){
        float dist = calcDistance(src, dst);
        //NSLog(@"%d %f min:%f\n", dst.getLabel(), dist, minDist);
        if(dist < minDist){
            minDist = dist;
            minLabel = dst.getLabel();
        }
    }
    
    return minLabel;
    
    
}


float Classifier::calcDistance(const Descriptor &src, const Descriptor& dst){
    


	//Scalar srcSum = cv::sum(srcFea);
	//Scalar dstSum = cv::sum(srcFea);
    
	//Mat sub = dstFea - srcFea;

	/*
	printf("sub start:");

	for (int i = 0; i < sub.cols * sub.rows; i++) {
		auto tmp = sub.at<float>(0, i);
		if (sub.at<float>(0, i) != 0) {
			printf("%f ", sub.at<float>(0, i));

		}

	}

	printf("sub end\n");*/
	//cv::Mat offsetSum = sub * sub.t();

    //float blobDist = cv::compareHist(src.getBlobFeature(), dst.getBlobFeature(), cv::HistCompMethods::HISTCMP_CHISQR);
    
    Mat srcFea = src.getHogFeature();
    Mat dstFea = dst.getHogFeature();
    
    cv::Mat hogSub = srcFea - dstFea;
    
    cv::Mat hogSubNorm = hogSub.t() * hogSub;
    
    float hogSubNormValue = hogSubNorm.at<float>(0, 0);
    /*
    float surfDist = 0;
    
    std::vector<cv::DMatch> keyMatches;
    
    m_feaMatcher->match(src.getSurfFeature(), dst.getSurfFeature(), keyMatches);
    
    sort(keyMatches.begin(), keyMatches.end(),
         [](const DMatch & a, const DMatch & b) -> bool
         {
             return a.distance < b.distance;
    });
    
    int ii = 0;
    
    for(const cv::DMatch& keyMatch: keyMatches){
        surfDist += keyMatch.distance;
        if(ii++ > 40){
            break;
        }
        
    }

	surfDist = surfDist / keyMatches.size();*/
    
    //float histDist = offsetSum.at<float>(0, 0);
    
    //offsetSum.release();
    srcFea.release();
    dstFea.release();
    hogSub.release();
    hogSubNorm.release();
    
    return hogSubNormValue;// * surfDist;//surfDist;
}


void Classifier::initSVMParams(const cv::Mat& rsp){
    /*
    vector<int> rspStat;
    int tot;
    
    for(int ii = 0; ii < m_nKind; ii++){
        
        int num = countNonZero(rsp == (ii+1));
        rspStat.push_back(num);
        tot+= num;
    }
    
    cv::Mat Ws(m_nKind, 1, CV_32FC1);
    
    for(int ii = 0; ii < m_nKind; ii++){
        Ws.at<float>(ii) = static_cast<float>(rspStat[ii])/ static_cast<float>(rspStat[ii]);
    }*/
    
    m_svm->setType(SVM::C_SVC);
    m_svm->setKernel(SVM::LINEAR);
    //m_svm->setClassWeights(Ws);
    m_svm->setC(1);
    
}

void Classifier::initGridParams(){
    
    c_grid = SVM::getDefaultGrid(SVM::C);
    gamma_grid = SVM::getDefaultGrid(SVM::GAMMA);
    
    p_grid = SVM::getDefaultGrid(SVM::P);
    p_grid.logStep = 0;
    
    nu_grid = SVM::getDefaultGrid(SVM::NU);
    nu_grid.logStep = 0;
    
    coef_grid = SVM::getDefaultGrid(SVM::COEF);
    coef_grid.logStep = 0;
    
    degree_grid = SVM::getDefaultGrid(SVM::DEGREE);
    degree_grid.logStep = 0;
    
    
    
}


void Classifier::svmTrain(cv::Mat& trainData, cv::Mat rsp, int nKind, std::string savePath){
    
    FileStorage svmSave(savePath, FileStorage::WRITE);
    m_svm = SVM::create();
    
    /*if(svmSave.isOpened()){
        
        m_svm = StatModel::load<SVM>(savePath);
        
    }else{
    */
    m_nKind = nKind;
    initSVMParams(rsp);
    
    m_svm->train(trainData, ROW_SAMPLE, rsp);
    
    float tmp = m_svm->predict(trainData.row(14));
    
    //cv::FileStorage fs(savePath, cv::FileStorage::WRITE);
    //m_svm->write(fs);
    
    
    m_svm->save(savePath);
    //}
   
    

    
    
}

void Classifier::load(std::string savePath){
    //FileStorage svmSave(savePath, FileStorage::READ);
    
    //auto svmStr = svmSave.releaseAndGetString();
    
    m_svm = cv::Algorithm::load<ml::SVM>(savePath);
    
}
     
int Classifier::svmPredict(cv::Mat& fea){
    
    cv::Mat res;
    
    //float val = m_svm->predict(fea, res);
    
    //return static_cast<int>(val + 0.5);
    return 1;
    
}


int Classifier::useless(cv::Mat& fea){
    float val = m_svm->predict(fea);
    return static_cast<int>(val);
}

























