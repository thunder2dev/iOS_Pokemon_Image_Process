//
//  Coder.cpp
//  MonsterS
//
//  Created by n01192 on 9/16/16.
//  Copyright © 2016 n01192. All rights reserved.
//

#include "Coder.hpp"

using namespace cv;

Coder::Coder()/*(cv::Mat& codebook)
:m_codebook(codebook),
m_flannIdx(cv::flann::Index(codebook, cv::flann::KDTreeIndexParams()))*/{
    
}

void Coder::setCodeBook(cv::Mat& codebook){
    m_codebook = codebook;
}



cv::Mat Coder::coeff2code(cv::Mat &coeff){
    
    cv::Mat code;
    
    cv::reduce(coeff, code, 0, cv::REDUCE_MAX);
    
    return code;
    
}


cv::Mat Coder::code(cv::Mat& input){
    
    
    cv::Mat idxMat = brutalFindKNN(m_codebook, input, 5);
    
    cv::Mat llcode = llccode(m_codebook, input, idxMat, 5);
    
    
    cv::Mat maxCode = coeff2code(llcode);
    
    cv::Mat normCode;
    cv::normalize(maxCode, normCode);
    
    double norm = cv::norm(normCode);
    
    return normCode;
    
    
}


cv::Mat Coder::llccode(cv::Mat &codebook, cv::Mat &input, cv::Mat IDX, int k)
{
    int nquery = input.rows;
    int nbase = codebook.rows;
    int dim = codebook.cols;
    
    Mat II = Mat::eye(k, k, CV_32FC1);
    Mat Coeff(nquery,nbase,CV_32FC1);
    Coeff.setTo(0);
    Mat z;
    Mat z1(k,dim,CV_32FC1);
    Mat z2(k,dim,CV_32FC1);
    Mat C;
    Mat un(k,1,CV_32FC1);
    un.setTo(1);
    Mat temp;
    Mat temp2;
    Mat w;
    Mat wt;
    
    for (int n = 0; n<nquery; n++) {
        for (int i = 0; i<k; i++) {
            for (int j = 0; j<dim; j++) {
                z1.at<float>(i,j) = codebook.at<float>(IDX.at<uchar>(n,i),j);
                z2.at<float>(i,j) = input.at<float>(n,j);
            }
        }
        z = z1 - z2;
        transpose(z, temp);
        C = z*temp;
        C = C + II*(1e-4)*trace(C)[0];
        invert(C,temp2);
        w = temp2*un;
        float sum_w=0;
        for (int i = 0; i<k; i++) {
            sum_w += w.at<float>(i,0);
        }
        w = w/sum_w;
        transpose(w, wt);
        for (int i = 0; i<k; i++) {
            Coeff.at<float>(n,IDX.at<uchar>(n,i)) = wt.at<float>(0,i);
        }
    }
    
    II.release();
    z.release();
    z1.release();
    z2.release();
    C.release();
    un.release();
    temp.release();
    temp2.release();
    w.release();
    wt.release();
    
    return Coeff;
}



cv::Mat Coder::brutalFindKNN(cv::Mat &codebook, cv::Mat &input, int k) {
    int nbase = codebook.rows;
    int nquery = input.rows;
    Mat ii = input.mul(input);
    Mat cc = codebook.mul(codebook);
    
    Mat sii(nquery,1,CV_32FC1);
    sii.setTo(0);
    Mat scc(nbase,1,CV_32FC1);
    scc.setTo(0);
    for (int i = 0; i<ii.rows; i++) {
        for (int j = 0; j<ii.cols; j++) {
            sii.at<float>(i,0) += ii.at<float>(i,j);
        }
    }
    
    
    for (int i = 0; i<cc.rows; i++) {
        for (int j = 0; j<cc.cols; j++) {
            scc.at<float>(i,0) += cc.at<float>(i,j);
        }
    }
    
    Mat D(nquery,nbase,CV_32FC1);
    for (int i = 0; i<nquery; i++) {
        for (int j = 0; j<nbase; j++) {
            D.at<float>(i,j) = sii.at<float>(i,0);
        }
    }
    
    Mat ct;
    transpose(codebook, ct);
    Mat D1 = 2*input*ct;
    
    Mat scct;
    transpose(scc, scct);
    Mat D2(nquery, nbase, CV_32FC1);
    for (int i = 0; i<nquery; i++) {
        for (int j = 0; j<nbase; j++) {
            D2.at<float>(i,j) = scct.at<float>(0,j);
        }
    }
    
    D = D - D1 + D2;
    Mat SD;
    sortIdx(D, SD, CV_SORT_EVERY_ROW+CV_SORT_ASCENDING);
    Mat IDX(nquery,k,CV_8UC1);
    for (int i = 0; i<nquery; i++) {
        for (int j = 0; j<k; j++) {
            IDX.at<uchar>(i,j) = SD.row(i).col(j).at<uchar>(0,0);
        }
    }
    
    ii.release();
    cc.release();
    sii.release();
    scc.release();
    D.release();
    ct.release();  
    D1.release();  
    scct.release();  
    D2.release();  
    SD.release();  
    return IDX;  
}






























